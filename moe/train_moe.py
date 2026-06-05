"""
GRPO training with Mixture-of-Experts on GSM8K.

Adapted from grpo/train_grpo.py.  The model must be pre-converted with
``moe/convert_dense.py`` before training.

Usage
-----
    # 1. Convert the dense model
    python -m moe.convert_dense --model_path Qwen/Qwen2.5-1.5B --output ./qwen_moe_init

    # 2. Train MoE with GRPO
    deepspeed --num_gpus=2 moe/train_moe.py \
        --deepspeed_config moe/ds_config_moe.json
"""

import argparse
import gc
import json
import math
import os
import re
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# When run as `deepspeed moe/train_moe.py`, Python adds moe/ to sys.path,
# not the project root.  Fix it so `from moe.moe_layer import ...` works.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

from moe.moe_layer import collect_aux_losses, convert_to_moe


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class PromptDataset(Dataset):
    def __init__(self, data, tokenizer, max_prompt_length=512):
        self.samples = []
        for example in data:
            prompt = f"Problem: {example['problem']}\nSolution:\n"
            encoded = tokenizer(prompt, truncation=True, max_length=max_prompt_length,
                                padding=False, return_tensors=None)
            gt = _extract_gt(example["solution"])
            self.samples.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "answer": gt,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Reward helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_answer(text: str) -> str | None:
    if "####" in text:
        return text.split("####")[-1].strip()
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m[-1].strip()
    return None


def _extract_gt(solution_text: str) -> str:
    ans = _extract_answer(solution_text)
    return ans if ans is not None else solution_text.strip()


def _normalize_number(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    if "/" in s and s.count("/") == 1:
        try:
            num, denom = s.split("/")
            s = f"{float(num) / float(denom):.10g}"
        except (ValueError, ZeroDivisionError):
            pass
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s = s[:-1]
    if s.startswith("."):
        s = "0" + s
    return s


def _gsm8k_reward(generated_text: str, ground_truth: str) -> float:
    pred = _extract_answer(generated_text)
    if pred is None:
        return 0.0
    return 1.0 if _normalize_number(pred) == _normalize_number(ground_truth) else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Collation
# ═══════════════════════════════════════════════════════════════════════════

def prompt_collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]
    max_len = max(ids.size(0) for ids in input_ids)
    padded_ids, padded_masks = [], []
    for ids in input_ids:
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            p = torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ids = torch.cat([p, ids], dim=0)
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids) - pad_len, dtype=torch.long),
            ])
        else:
            mask = torch.ones(len(ids), dtype=torch.long)
        padded_ids.append(ids)
        padded_masks.append(mask)
    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_masks),
        "answers": [s["answer"] for s in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Logprobs
# ═══════════════════════════════════════════════════════════════════════════

def _sequence_logprobs(model, input_ids, attention_mask, micro_batch=2):
    B = input_ids.size(0)
    results = []
    for start in range(0, B, micro_batch):
        end = min(start + micro_batch, B)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_ids = chunk_ids[:, 1:]
        token_logprobs = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_ids.reshape(-1),
            reduction="none",
        ).reshape_as(shift_ids).float()
        results.append(token_logprobs)
    return torch.cat(results, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# GRPO loss  (same formulation as grpo/train_grpo.py)
# ═══════════════════════════════════════════════════════════════════════════

def grpo_loss(new_logprobs, old_logprobs, ref_logprobs, advantages, response_mask,
              clip_epsilon=0.2, kl_coef=0.04):
    ratio_old = torch.exp(new_logprobs - old_logprobs)
    adv = advantages.unsqueeze(-1).expand_as(ratio_old)
    surr1 = ratio_old * adv
    surr2 = torch.clamp(ratio_old, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    policy_loss = -torch.min(surr1, surr2)

    mask = response_mask.float()
    n_tokens = mask.sum().clamp(min=1)

    kl_per_token = new_logprobs - ref_logprobs
    kl_per_token = kl_per_token + torch.exp(-kl_per_token) - 1.0

    loss = ((policy_loss * mask).sum()
            + kl_coef * torch.clamp((kl_per_token * mask).sum(), min=0.0)) / n_tokens

    with torch.no_grad():
        clipped_frac = ((ratio_old < 1.0 - clip_epsilon) | (ratio_old > 1.0 + clip_epsilon)).float()
        stats = {
            "policy_loss": (policy_loss * mask).sum() / n_tokens,
            "kl_divergence": (kl_per_token * mask).sum() / n_tokens,
            "clipped_fraction": (clipped_frac * mask).sum() / n_tokens,
        }
    return loss, stats


# ═══════════════════════════════════════════════════════════════════════════
# Rollout generation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_rollouts_batched(model, tokenizer, prompt_ids, prompt_mask,
                               group_size, max_new_tokens, temperature, top_p):
    B, L = prompt_ids.shape
    model_device = next(model.parameters()).device
    batch_ids = prompt_ids.repeat_interleave(group_size, dim=0).to(model_device)
    batch_mask = prompt_mask.repeat_interleave(group_size, dim=0).to(model_device)
    gens = model.generate(
        input_ids=batch_ids, attention_mask=batch_mask,
        max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, use_cache=True,
    )
    results = []
    for i in range(B * group_size):
        full_ids = gens[i].cpu()
        resp_mask = torch.zeros(full_ids.size(0), dtype=torch.bool)
        resp_mask[L:] = True
        prompt_len = int(prompt_mask[i // group_size].sum().item())
        results.append({"full_ids": full_ids, "response_mask": resp_mask,
                        "prompt_len": prompt_len})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Padding helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pad_rollouts(rollouts, tokenizer):
    ids_list = [r["full_ids"] for r in rollouts]
    mask_list = [r["response_mask"] for r in rollouts]
    max_len = max(ids.size(0) for ids in ids_list)
    N = len(ids_list)
    padded_ids = torch.full((N, max_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_mask = torch.zeros(N, max_len, dtype=torch.bool)
    for i, (ids, m) in enumerate(zip(ids_list, mask_list)):
        L = ids.size(0)
        padded_ids[i, max_len - L:] = ids
        padded_mask[i, max_len - L:] = m
    return padded_ids, padded_mask


def _slice_response(full_tensor, full_resp_mask):
    B = full_tensor.size(0)
    mask_shifted = full_resp_mask[:, 1:].bool().to(full_tensor.device)
    counts = mask_shifted.sum(dim=1)
    max_rl = counts.max().item()
    if max_rl == 0:
        out = torch.zeros(B, 1, dtype=full_tensor.dtype, device=full_tensor.device)
        out_mask = torch.zeros(B, 1, dtype=full_tensor.dtype, device=full_tensor.device)
        return out, out_mask
    out = torch.zeros(B, max_rl, dtype=full_tensor.dtype, device=full_tensor.device)
    out_mask = torch.zeros(B, max_rl, dtype=full_tensor.dtype, device=full_tensor.device)
    for b in range(B):
        c = counts[b].item()
        if c > 0:
            out[b, :c] = full_tensor[b][mask_shifted[b]]
            out_mask[b, :c] = 1.0
    return out, out_mask


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _evaluate_gsm8k_test(model_engine, tokenizer, max_new=256, max_samples=200,
                          eval_batch_size=8):
    model_engine.module.eval()
    test_set = load_dataset("openai/gsm8k", "main", split="test")
    prompts, gt_answers = [], []
    for example in test_set:
        prompts.append(f"Problem: {example['question']}\nSolution:\n")
        sol = example["answer"]
        gt = _extract_answer(sol)
        gt_answers.append(gt if gt is not None else sol.strip())

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = model_engine.device
    correct, total = 0, 0
    max_per_rank = min(max_samples, len(prompts)) // dist.get_world_size()

    for i in range(0, min(max_samples, len(prompts)), eval_batch_size):
        if total >= max_per_rank:
            break
        batch_prompts = prompts[i:i + eval_batch_size]
        batch_gt = gt_answers[i:i + eval_batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=768)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].size(1)
        gens = model_engine.module.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True,
        )
        for gen, gt in zip(gens, batch_gt):
            response_ids = gen[input_len:]
            decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
            correct += int(_gsm8k_reward(decoded, gt))
            total += 1
            if total >= max_per_rank:
                break

    tokenizer.padding_side = orig_side
    model_engine.module.train()
    return correct / max(total, 1)


def _all_reduce_scalar(value, device):
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--model_path", type=str, default="moe/qwen_moe_init")
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--group_size", type=int, default=4)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.08)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--per_device_prompt_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="Batch size for GSM8K evaluation (default: 16).")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="moe/moe_qwen_gsm8k")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--balance_alpha", type=float, default=0.01,
                   help="Weight of MoE load-balancing auxiliary loss")
    p.add_argument("--z_loss_beta", type=float, default=0.001,
                   help="Weight of router Z-loss")
    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _mem(msg: str, rank: int):
    """Log CPU memory usage if psutil is available."""
    try:
        import psutil
        p = psutil.Process()
        rss = p.memory_info().rss / (1024 ** 3)
        print(f"[rank{rank}] MEM {msg}: {rss:.2f} GB RSS")
    except ImportError:
        pass


def train():
    args = parse_args()
    deepspeed.init_distributed()
    rank = dist.get_rank()

    _mem("after init_distributed", rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load MoE conversion config ────────────────────────────────────
    moe_config_path = os.path.join(args.model_path, "moe_config.json")
    if not os.path.isfile(moe_config_path):
        raise FileNotFoundError(
            f"{moe_config_path} not found. Run moe/convert_dense.py first."
        )
    with open(moe_config_path) as f:
        moe_config = json.load(f)

    # ── Dataset ──────────────────────────────────────────────────────
    # CRITICAL: create DataLoader BEFORE loading models.  DataLoader workers
    # are forked from the parent process.  If the parent has ~8.5 GB of GPU
    # tensor metadata, each fork inherits that address space — tripling
    # virtual memory and triggering OOM on a 32 GB machine.
    raw = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    raw = raw.filter(lambda x: x["source"] == "gsm8k")
    train_data = PromptDataset(raw, tokenizer, args.max_prompt_length)
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data, batch_size=args.per_device_prompt_batch_size,
        sampler=train_sampler, num_workers=0,
        # num_workers=0 avoids fork entirely — data is loaded in the main
        # process, trading a small throughput cost for a large memory saving.
        collate_fn=lambda b: prompt_collate_fn(b, tokenizer),
    )

    _mem("after dataset + dataloader", rank)

    # ── Models ───────────────────────────────────────────────────────
    # Load to CPU — DeepSpeed ZeRO-2 owns device placement.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    _mem("after dense model load", rank)

    model.config.use_cache = False
    model = convert_to_moe(model, moe_config)
    _mem("after moe conversion", rank)

    model.gradient_checkpointing_enable()
    model.train()

    # Reference model stays dense, also on CPU — DeepSpeed will move it
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    _mem("after ref model load", rank)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    # ref_model stays on CPU except during its forward pass (step 4).

    # ── DeepSpeed ────────────────────────────────────────────────────
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    _mem("after deepspeed init", rank)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024**3)
            reserv = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"[rank{rank}] GPU{i} after deepspeed init: "
                  f"{alloc:.2f} GB allocated, {reserv:.2f} GB reserved")

    # ── Scheduler ───────────────────────────────────────────────────
    total_steps = len(train_loader) * args.num_epochs
    base_opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    scheduler = get_linear_schedule_with_warmup(
        base_opt, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    writer = SummaryWriter(os.path.join(args.output_dir, "logs")) if rank == 0 else None

    # ── Resume ───────────────────────────────────────────────────────
    resume_step = 0
    start_epoch = 0
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint.rstrip("/")
        parent, last = os.path.split(ckpt_path)
        if last.startswith("step_") or last in ("final", "latest"):
            load_dir, tag = parent, last
        elif last.startswith("epoch_"):
            load_dir, tag = parent, last
        else:
            load_dir, tag = ckpt_path, None
        if rank == 0:
            print(f"Loading checkpoint: load_dir={load_dir}, tag={tag or '(latest)'}")
        _, client_state = model_engine.load_checkpoint(load_dir, tag=tag)
        if client_state is None:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}")
        resume_step = client_state.get("global_step", 0)
        start_epoch = resume_step // len(train_loader)
        if scheduler is not None:
            scheduler.last_epoch = resume_step
        if rank == 0:
            print(f"Resumed at step {resume_step}, epoch {start_epoch + 1}")

    # ── Training ────────────────────────────────────────────────────
    global_step = resume_step
    ds_device = model_engine.device

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        n_skip = resume_step % len(train_loader) if epoch == start_epoch else 0
        skipped = 0

        for batch in train_loader:
            if skipped < n_skip:
                skipped += 1
                continue

            prompt_ids = batch["input_ids"].to(ds_device)
            prompt_mask = batch["attention_mask"].to(ds_device)
            answers = batch["answers"]
            B = prompt_ids.size(0)
            G = args.group_size

            # ── 1. Generate rollouts ─────────────────────────────
            # ZeRO-2 keeps full params on each GPU — no all-gather
            # needed, generate() works directly.
            ref_model = ref_model.to("cpu")
            torch.cuda.empty_cache()

            if rank == 0:
                print(f"[Step {global_step}] Generating rollouts...")
            model_engine.module.eval()
            rollouts = _generate_rollouts_batched(
                model_engine.module, tokenizer,
                prompt_ids, prompt_mask, group_size=G,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            model_engine.module.train()
            for r in rollouts:
                r["full_ids"] = r["full_ids"].cpu()
                r["response_mask"] = r["response_mask"].cpu()

            all_rewards = torch.zeros(B * G, device=ds_device)
            for j, r in enumerate(rollouts):
                decoded = tokenizer.decode(r["full_ids"], skip_special_tokens=True)
                all_rewards[j] = _gsm8k_reward(decoded, answers[j // G])

            # Bring ref model back for forward passes
            ref_model = ref_model.to(ds_device)

            # ── 2. Advantages ────────────────────────────────────
            rewards_2d = all_rewards.view(B, G)
            mean_r = rewards_2d.mean(dim=-1, keepdim=True)
            std_r = rewards_2d.std(dim=-1, keepdim=True) + 1e-8
            advantages = ((rewards_2d - mean_r) / std_r).view(-1)

            # ── 3. Pad ───────────────────────────────────────────
            full_ids, full_resp_masks = _pad_rollouts(rollouts, tokenizer)
            model_device = next(model_engine.module.parameters()).device
            full_ids = full_ids.to(model_device)
            full_resp_masks = full_resp_masks.to(model_device)
            attn_mask = (full_ids != tokenizer.pad_token_id).long()

            # ── 4. Forward passes ────────────────────────────────
            if rank == 0:
                print(f"[Step {global_step}] old logprobs...")
            with torch.no_grad():
                old_full = _sequence_logprobs(model_engine.module, full_ids, attn_mask)
                if rank == 0:
                    print(f"[Step {global_step}] ref logprobs...")
                ref_full = _sequence_logprobs(ref_model, full_ids, attn_mask)
            if rank == 0:
                print(f"[Step {global_step}] slicing...")
            old_logprobs, _ = _slice_response(old_full, full_resp_masks)
            ref_logprobs, _ = _slice_response(ref_full, full_resp_masks)
            del old_full, ref_full

            # Offload ref model back to CPU — frees ~3 GB
            if rank == 0:
                print(f"[Step {global_step}] offloading ref...")
            ref_model = ref_model.to("cpu")
            torch.cuda.empty_cache()

            if rank == 0:
                print(f"[Step {global_step}] new logprobs...")
            new_full = _sequence_logprobs(model_engine.module, full_ids, attn_mask)
            if rank == 0:
                print(f"[Step {global_step}] loss...")
            new_logprobs, resp_mask = _slice_response(new_full, full_resp_masks)

            # ── 5. Loss ──────────────────────────────────────────
            grpo_l, stats = grpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs.detach(),
                ref_logprobs=ref_logprobs.detach(),
                advantages=advantages,
                response_mask=resp_mask,
                clip_epsilon=args.clip_epsilon,
                kl_coef=args.kl_coef,
            )

            # Add MoE auxiliary losses
            moe_aux = collect_aux_losses(model_engine.module)
            aux_loss = (args.balance_alpha * moe_aux["balance_loss"]
                        + args.z_loss_beta * moe_aux["z_loss"])
            loss = grpo_l + aux_loss

            if rank == 0:
                print(f"[Step {global_step}] backward...")
            model_engine.backward(loss)
            if rank == 0:
                print(f"[Step {global_step}] step...")
            model_engine.step()
            if scheduler is not None and model_engine.is_gradient_accumulation_boundary():
                scheduler.step()
            global_step += 1

            # ── 6. Logging ──────────────────────────────────────
            if rank == 0 and global_step % args.logging_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/grpo_loss", grpo_l.item(), global_step)
                writer.add_scalar("train/mean_reward", all_rewards.mean().item(), global_step)
                writer.add_scalar("train/kl", stats["kl_divergence"].item(), global_step)
                writer.add_scalar("moe/balance_loss", moe_aux["balance_loss"].item(), global_step)
                writer.add_scalar("moe/z_loss", moe_aux["z_loss"].item(), global_step)
                writer.add_scalar("moe/dropped_frac", moe_aux["dropped_frac"].item(), global_step)
                print(
                    f"[Epoch {epoch+1}] Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reward: {all_rewards.mean().item():.3f} | "
                    f"KL: {stats['kl_divergence'].item():.4f} | "
                    f"Bal: {moe_aux['balance_loss'].item():.4f}"
                )

            if all_rewards.mean().item() == 0.0 and rank == 0:
                print(f"  ⚠ All rewards zero at step {global_step}. Samples:")
                for k in range(min(3, len(rollouts))):
                    sample = tokenizer.decode(rollouts[k]["full_ids"], skip_special_tokens=True)
                    resp_len = rollouts[k]["full_ids"].size(0) - rollouts[k]["prompt_len"]
                    print(f"    [{resp_len} tokens] {sample[-250:]}")

            # ── 7. Eval ─────────────────────────────────────────
            if global_step % args.eval_steps == 0:
                local_acc = _evaluate_gsm8k_test(model_engine, tokenizer,
                                                  max_new=args.max_new_tokens,
                                                  eval_batch_size=args.eval_batch_size)
                global_acc = _all_reduce_scalar(local_acc, device=model_engine.device)
                if rank == 0:
                    writer.add_scalar("eval/gsm8k_accuracy", global_acc, global_step)
                    print(f"[Eval]  Step {global_step} | GSM8K Accuracy: {global_acc:.4f}")

            # ── 8. Save ─────────────────────────────────────────
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                model_engine.save_checkpoint(
                    args.output_dir, tag=f"step_{global_step}",
                    client_state={"global_step": global_step},
                )

        # End of epoch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        model_engine.save_checkpoint(
            args.output_dir, tag=f"epoch_{epoch+1}",
            client_state={"global_step": global_step},
        )

    model_engine.save_checkpoint(args.output_dir, tag="final",
                                 client_state={"global_step": global_step})
    if rank == 0:
        writer.close()
        print("Training complete.")


if __name__ == "__main__":
    train()
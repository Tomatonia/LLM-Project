"""
GRPO (Group Relative Policy Optimization) training for mathematical reasoning.

Architecture
------------
GRPO is a reinforcement learning method that fine-tunes LLMs without a critic
network. For each prompt, it samples a *group* of G responses, scores them with
a reward function, then normalizes rewards *within the group* to compute
advantages.  The policy is updated with a clipped surrogate objective, plus a
KL penalty against a frozen reference model.

Pipeline per training step
--------------------------
1. Sample B prompts from GSM8K (distributed across GPUs via DistributedSampler).
2. For each prompt, generate G = 4 candidate solutions (sampling, temp=1.0).
3. Extract the final answer from each solution; compare with ground truth.
   Reward = 1.0 (correct) or 0.0 (incorrect).
4. Group-relative advantage:
      A_i = (r_i - mean(r_group)) / (std(r_group) + 1e-8)
5. Forward the concatenated [prompt + response] sequences through:
   - The trainable policy model  (no-grad) → old_logprobs (ratio denominator)
   - The frozen reference model  (no-grad) → ref_logprobs (KL baseline)
   - The trainable policy model  (w/ grad) → new_logprobs (ratio numerator)
6. GRPO loss (per token, response tokens only):
      ratio = exp(log π_θ - log π_old)
      L_clip = min(ratio · A, clip(ratio, 0.8, 1.2) · A)
      L_KL = log π_θ - log π_ref
      L = -mean(L_clip) + β · mean(L_KL)
7. Backward pass + optimizer step via DeepSpeed.

Usage
-----
    deepspeed --num_gpus=2 grpo/train_grpo.py \
        --model_path ./sft_qwen_gsm8k \
        --output_dir ./grpo_qwen_gsm8k \
        --num_epochs 2 \
        --group_size 4 \
        --kl_coef 0.04
"""

import argparse
import gc
import os
import re
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class PromptDataset(Dataset):
    """Tokenized prompts with ground-truth answers for GSM8K."""

    def __init__(self, data, tokenizer, max_prompt_length: int = 512):
        self.samples = []
        for example in data:
            problem = example["problem"]
            prompt = f"Problem: {problem}\nSolution:\n"
            encoded = tokenizer(
                prompt, truncation=True, max_length=max_prompt_length,
                padding=False, return_tensors=None,
            )
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
    """Extract the final answer from a solution, trying #### then \\boxed{}."""
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
# Collation helpers
# ═══════════════════════════════════════════════════════════════════════════

def prompt_collate_fn(batch, tokenizer):
    """Left-pad prompts so generation starts at the rightmost position."""
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
# Log-probability computation  (NO decorator — caller controls grad)
# ═══════════════════════════════════════════════════════════════════════════

def _sequence_logprobs(model, input_ids, attention_mask, micro_batch=4):
    """
    Token-level log-probabilities for a batch, processed in micro-batches
    to limit peak GPU memory from the logits tensor (B, L, V).

    Returns (B, L-1) in float32 where position i holds
    log P(token_{i+1} | token_{0..i}).
    """
    B = input_ids.size(0)
    results = []
    for start in range(0, B, micro_batch):
        end = min(start + micro_batch, B)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask)
        logits = outputs.logits                                    # (mb, L, V)  bfloat16
        shift_logits = logits[:, :-1, :].contiguous()              # (mb, L-1, V)
        shift_ids = chunk_ids[:, 1:]                               # (mb, L-1)

        token_logprobs = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_ids.reshape(-1),
            reduction="none",
        ).reshape_as(shift_ids).float()                            # (mb, L-1)  float32
        results.append(token_logprobs)

    return torch.cat(results, dim=0)                               # (B, L-1)


# ═══════════════════════════════════════════════════════════════════════════
# GRPO loss
# ═══════════════════════════════════════════════════════════════════════════

def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.04,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO clipped surrogate + KL penalty.

    All logprob / mask tensors are (B, L_resp) — already sliced to response
    tokens only, right-padded with zeros for shorter sequences.

    advantages: (B,) per-response scalar advantage.
    """
    # Importance ratio  (B, L_resp)
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Broadcast per-response advantage to each token
    adv = advantages.unsqueeze(-1).expand_as(ratio)

    # PPO-style clipped surrogate  (we minimise, so negate)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    policy_loss = -torch.min(surr1, surr2)

    # KL penalty  (log π_θ - log π_ref), clamped to ≥ 0 per token
    # Negative values are estimation noise: tokens were sampled from π_old,
    # not π_θ, so new_logprobs can legitimately be < ref_logprobs per token.
    kl_per_token = torch.clamp(new_logprobs - ref_logprobs, min=0.0)

    mask = response_mask.float()
    n_tokens = mask.sum().clamp(min=1)

    loss = ((policy_loss * mask).sum() + kl_coef * (kl_per_token * mask).sum()) / n_tokens

    with torch.no_grad():
        clipped_frac = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float()
        # Report the unclamped estimate for monitoring
        raw_kl = (new_logprobs - ref_logprobs) * mask
        stats = {
            "policy_loss": (policy_loss * mask).sum() / n_tokens,
            "kl_divergence": raw_kl.sum() / n_tokens,
            "clipped_fraction": (clipped_frac * mask).sum() / n_tokens,
        }
    return loss, stats


# ═══════════════════════════════════════════════════════════════════════════
# Rollout generation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_rollouts_batched(model, tokenizer, prompt_ids, prompt_mask,
                               group_size, max_new_tokens, temperature, top_p):
    """
    Generate G responses for each of B prompts in a single batched call.

    *prompt_ids* / *prompt_mask* are (B, L) already left-padded to the same
    length.  Returns a flat list of B*G rollout dicts.
    """
    B, L = prompt_ids.shape
    model_device = next(model.parameters()).device

    # Repeat each prompt G times → (B*G, L)
    batch_ids = prompt_ids.repeat_interleave(group_size, dim=0).to(model_device)
    batch_mask = prompt_mask.repeat_interleave(group_size, dim=0).to(model_device)

    gens = model.generate(
        input_ids=batch_ids,
        attention_mask=batch_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    results = []
    for i in range(B * group_size):
        full_ids = gens[i].cpu()
        # full_ids = [left_padding | actual_prompt | response]
        # L is the padded input length; everything after L is response
        resp_mask = torch.zeros(full_ids.size(0), dtype=torch.bool)
        resp_mask[L:] = True
        prompt_len = int(prompt_mask[i // group_size].sum().item())
        results.append({
            "full_ids": full_ids,
            "response_mask": resp_mask,
            "prompt_len": prompt_len,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Batch padding & response extraction
# ═══════════════════════════════════════════════════════════════════════════

def _pad_rollouts(rollouts: list[dict], tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad rollout dicts into (N, L_max) tensors."""
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


def _slice_response(full_tensor: torch.Tensor, full_resp_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract response-only tokens from a full-sequence tensor.

    full_tensor   : (B, L-1) — logit-aligned (already shifted left by 1)
    full_resp_mask: (B, L)   — per-token bool mask (True = response)

    Returns (B, max_r) right-padded with zeros, plus a mask of the same shape.
    """
    B = full_tensor.size(0)
    mask_shifted = full_resp_mask[:, 1:].bool().to(full_tensor.device)   # (B, L-1)
    counts = mask_shifted.sum(dim=1)
    max_rl = counts.max().item()

    if max_rl == 0:
        return torch.zeros(B, 1, dtype=full_tensor.dtype, device=full_tensor.device), \
               torch.zeros(B, 1, dtype=full_tensor.dtype, device=full_tensor.device)

    out = torch.zeros(B, max_rl, dtype=full_tensor.dtype, device=full_tensor.device)
    out_mask = torch.zeros(B, max_rl, dtype=full_tensor.dtype, device=full_tensor.device)

    for b in range(B):
        c = counts[b].item()
        if c > 0:
            out[b, :c] = full_tensor[b][mask_shifted[b]]
            out_mask[b, :c] = 1.0

    return out, out_mask


# ═══════════════════════════════════════════════════════════════════════════
# Distributed evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _evaluate_gsm8k_test(model_engine, tokenizer, max_new=256,
                          max_samples=200, eval_batch_size=8):
    """
    Evaluate accuracy on the official GSM8K test set (openai/gsm8k).

    Uses the same answer extraction as the standalone eval script,
    supporting both ``####`` and ``\\boxed{}`` formats.
    """
    model_engine.module.eval()
    test_set = load_dataset("openai/gsm8k", "main", split="test")

    # Build prompts and ground-truth answers
    prompts = []
    gt_answers = []
    for example in test_set:
        prompts.append(f"Problem: {example['question']}\nSolution:\n")
        sol = example["answer"]
        gt = _extract_answer(sol)
        gt_answers.append(gt if gt is not None else sol.strip())

    # Left-pad for batched generation
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
            **inputs,
            max_new_tokens=max_new, do_sample=False,
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


def _all_reduce_scalar(value: float, device) -> float:
    t = torch.tensor(value, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing & main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="GRPO fine-tuning on GSM8K")
    p.add_argument("--local_rank", type=int, default=-1)

    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B",
                   help="Path to SFT checkpoint or base model")
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # GRPO hyper-parameters
    p.add_argument("--group_size", type=int, default=4,
                   help="G: responses sampled per prompt")
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.04)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)

    # Training
    p.add_argument("--per_device_prompt_batch_size", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--warmup_steps", type=int, default=20)

    # Logging & saving
    p.add_argument("--output_dir", type=str, default="./grpo_qwen_gsm8k")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)

    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def train():
    args = parse_args()
    deepspeed.init_distributed()
    rank = dist.get_rank()
    device = torch.cuda.current_device()

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────
    raw = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    raw = raw.filter(lambda x: x["source"] == "gsm8k")
    train_data = PromptDataset(raw, tokenizer, args.max_prompt_length)

    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.per_device_prompt_batch_size,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=lambda b: prompt_collate_fn(b, tokenizer),
        pin_memory=True,
    )

    # ── Policy model (trainable) ────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    # ── Reference model (frozen, KL baseline) ───────────────────────
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── DeepSpeed engine ────────────────────────────────────────────
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # Place reference model on the same device as the policy engine
    ref_model = ref_model.to(model_engine.device)

    # ── Scheduler ───────────────────────────────────────────────────
    total_steps = len(train_loader) * args.num_epochs
    from transformers import get_linear_schedule_with_warmup
    base_opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    scheduler = get_linear_schedule_with_warmup(
        base_opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    # ── TensorBoard ─────────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(args.output_dir, "logs")) if rank == 0 else None

    # ── Training ────────────────────────────────────────────────────
    global_step = 0
    ds_device = model_engine.device

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            prompt_ids = batch["input_ids"].to(ds_device)
            prompt_mask = batch["attention_mask"].to(ds_device)
            answers = batch["answers"]
            B = prompt_ids.size(0)
            G = args.group_size

            # ── 1. Generate rollouts ─────────────────────────────
            # Eval mode disables gradient checkpointing so KV-cache works
            model_engine.module.eval()
            rollouts = _generate_rollouts_batched(
                model_engine.module, tokenizer,
                prompt_ids, prompt_mask,
                group_size=G,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            all_rewards = torch.zeros(B * G, device=ds_device)
            for j, r in enumerate(rollouts):
                decoded = tokenizer.decode(r["full_ids"], skip_special_tokens=True)
                all_rewards[j] = _gsm8k_reward(decoded, answers[j // G])

            # ── 2. Group-relative advantages ────────────────────
            model_engine.module.train()
            rewards_2d = all_rewards.view(B, G)
            mean_r = rewards_2d.mean(dim=-1, keepdim=True)
            std_r  = rewards_2d.std(dim=-1, keepdim=True) + 1e-8
            advantages = ((rewards_2d - mean_r) / std_r).view(-1)      # (B*G,)

            # ── 3. Pad sequences ────────────────────────────────
            full_ids, full_resp_masks = _pad_rollouts(rollouts, tokenizer)
            # Place tensors on the same device as the policy model
            model_device = next(model_engine.module.parameters()).device
            full_ids = full_ids.to(model_device)
            full_resp_masks = full_resp_masks.to(model_device)
            attn_mask = (full_ids != tokenizer.pad_token_id).long()

            # ── 4. Forward passes ───────────────────────────────
            # 4a. Old policy logprobs and reference logprobs (no grad)
            with torch.no_grad():
                old_full = _sequence_logprobs(model_engine.module, full_ids, attn_mask)
                ref_full = _sequence_logprobs(ref_model, full_ids, attn_mask)

            # 4b. Slice to response-only tokens, then free full tensors
            old_logprobs, _ = _slice_response(old_full, full_resp_masks)
            ref_logprobs, _ = _slice_response(ref_full, full_resp_masks)
            del old_full, ref_full

            # 4c. New policy logprobs (WITH grad — ratio numerator)
            new_full = _sequence_logprobs(model_engine.module, full_ids, attn_mask)
            new_logprobs, resp_mask = _slice_response(new_full, full_resp_masks)

            # ── 5. GRPO loss ────────────────────────────────────
            loss, stats = grpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs.detach(),
                ref_logprobs=ref_logprobs.detach(),
                advantages=advantages,
                response_mask=resp_mask,
                clip_epsilon=args.clip_epsilon,
                kl_coef=args.kl_coef,
            )

            model_engine.backward(loss)
            model_engine.step()
            if scheduler is not None and model_engine.is_gradient_accumulation_boundary():
                scheduler.step()
            global_step += 1

            # ── 6. Logging ──────────────────────────────────────
            if rank == 0 and global_step % args.logging_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/mean_reward", all_rewards.mean().item(), global_step)
                writer.add_scalar("train/kl", stats["kl_divergence"].item(), global_step)
                writer.add_scalar("train/clip_frac", stats["clipped_fraction"].item(), global_step)
                print(
                    f"[Epoch {epoch+1}] Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reward: {all_rewards.mean().item():.3f} | "
                    f"KL: {stats['kl_divergence'].item():.4f}"
                )

            # ── 6b. Collapse detection ──────────────────────────
            if all_rewards.mean().item() == 0.0 and rank == 0:
                print(f"  ⚠ All rewards zero at step {global_step}. Sample generations:")
                for k in range(min(3, len(rollouts))):
                    sample = tokenizer.decode(rollouts[k]["full_ids"], skip_special_tokens=True)
                    resp_len = rollouts[k]["full_ids"].size(0) - rollouts[k]["prompt_len"]
                    print(f"    [{resp_len} tokens] {sample[-250:]}")
                # Check for NaN in model parameters
                has_nan = any(
                    torch.isnan(p).any().item()
                    for p in model_engine.module.parameters()
                    if p.requires_grad
                )
                if has_nan:
                    print(f"  ❌ NaN detected in model parameters at step {global_step}!")

            # ── 7. Eval ─────────────────────────────────────────
            if global_step % args.eval_steps == 0:
                local_acc = _evaluate_gsm8k_test(model_engine, tokenizer,
                                                  max_new=args.max_new_tokens)
                global_acc = _all_reduce_scalar(local_acc, device=model_engine.device)
                if rank == 0:
                    writer.add_scalar("eval/gsm8k_accuracy", global_acc, global_step)
                    print(f"[Eval]  Step {global_step} | GSM8K Accuracy: {global_acc:.4f}")

            # ── 8. Save checkpoint ──────────────────────────────
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

    # Final save
    model_engine.save_checkpoint(args.output_dir, tag="final",
                                 client_state={"global_step": global_step})
    if rank == 0:
        writer.close()
        print("Training complete.")


if __name__ == "__main__":
    train()

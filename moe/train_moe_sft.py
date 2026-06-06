"""
Stage 1: SFT training of MoE layers with frozen attention and non-MoE FFN.

Loads a dense model pre-converted with ``moe/convert_dense.py``, replaces the
target layers with MoE blocks, then freezes all parameters except the MoE
router + experts.  Trains with standard cross-entropy SFT loss on GSM8K.

After this stage, the MoE router and experts have learned the task, while the
attention and non-MoE FFN weights are unchanged from the dense pretrained model.

Usage
-----
    # 1. Convert the dense model
    python -m moe.convert_dense --model_path Qwen/Qwen2.5-1.5B --output ./qwen_moe_init

    # 2. Stage 1: SFT (frozen attention + non-MoE FFN)
    deepspeed --num_gpus=2 moe/train_moe_sft.py \
        --deepspeed_config moe/ds_config_moe.json

    # 3. Stage 2: GRPO (all layers unfrozen, resume from Stage 1)
    deepspeed --num_gpus=2 moe/train_moe.py \
        --model_path moe/qwen_moe_init \
        --resume_from_checkpoint moe/moe_sft_gsm8k/final \
        --resume_step 0 \
        --no-load_optimizer_states \
        --deepspeed_config moe/ds_config_moe.json
"""

import argparse
import gc
import json
import math
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

from moe.moe_layer import collect_aux_losses, convert_to_moe


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """Formats GSM8K problems as prompt+solution with prompt-token masking."""

    def __init__(self, data, tokenizer, max_length=1024):
        self.samples = []
        for example in data:
            full_text = f"Problem: {example['problem']}\nSolution:\n{example['solution']}"
            prompt_text = f"Problem: {example['problem']}\nSolution:\n"

            full_tokenized = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )
            prompt_tokenized = tokenizer(
                prompt_text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )

            input_ids = full_tokenized["input_ids"]
            prompt_ids = prompt_tokenized["input_ids"]
            prompt_len = len(prompt_ids)

            # Verify prefix consistency (BPE tokenizer should be deterministic)
            if input_ids[:prompt_len] != prompt_ids:
                for k in range(min(len(prompt_ids), len(input_ids))):
                    if input_ids[k] != prompt_ids[k]:
                        prompt_len = k
                        break

            labels = input_ids.copy()
            labels[:prompt_len] = [-100] * prompt_len

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": full_tokenized["attention_mask"],
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Collation
# ═══════════════════════════════════════════════════════════════════════════

def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(s["input_ids"], dtype=torch.long) for s in batch]
    attention_mask = [torch.tensor(s["attention_mask"], dtype=torch.long) for s in batch]
    labels = [torch.tensor(s["labels"], dtype=torch.long) for s in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ═══════════════════════════════════════════════════════════════════════════
# Layer freezing
# ═══════════════════════════════════════════════════════════════════════════

def freeze_non_moe_layers(model, moe_config):
    """Freeze attention, non-MoE FFN, embeddings, lm_head, and layer norms.

    Only the MoE routers + experts (shared and routed) remain trainable.
    """
    moe_layer_indices = set(moe_config["moe_layers"])

    # Embeddings and output head
    for p in model.model.embed_tokens.parameters():
        p.requires_grad_(False)
    for p in model.lm_head.parameters():
        p.requires_grad_(False)
    for p in model.model.norm.parameters():
        p.requires_grad_(False)

    for i, layer in enumerate(model.model.layers):
        # Freeze attention in all layers
        for p in layer.self_attn.parameters():
            p.requires_grad_(False)

        # Freeze layer norms
        for p in layer.input_layernorm.parameters():
            p.requires_grad_(False)
        for p in layer.post_attention_layernorm.parameters():
            p.requires_grad_(False)

        # Freeze non-MoE MLP (dense FFN layers that weren't converted)
        if i not in moe_layer_indices:
            for p in layer.mlp.parameters():
                p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen non-MoE layers. "
          f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model_engine, val_loader, tokenizer):
    model_engine.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in val_loader:
        batch = {k: v.to(model_engine.device) for k, v in batch.items()}
        outputs = model_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    loss_t = torch.tensor(total_loss, device=model_engine.device)
    count_t = torch.tensor(total_tokens, device=model_engine.device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

    avg_loss = (loss_t / count_t).item()
    model_engine.train()
    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--model_path", type=str, default="moe/qwen_moe_init")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="moe/moe_sft_gsm8k")
    p.add_argument("--balance_alpha", type=float, default=0.01,
                   help="Weight of MoE load-balancing auxiliary loss")
    p.add_argument("--z_loss_beta", type=float, default=0.001,
                   help="Weight of router Z-loss")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def train():
    args = parse_args()
    deepspeed.init_distributed()
    rank = dist.get_rank()

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
    raw = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    raw = raw.filter(lambda x: x["source"] == "gsm8k")
    raw = raw.train_test_split(test_size=0.1, seed=42)
    train_data = SFTDataset(raw["train"], tokenizer, args.max_length)
    val_data = SFTDataset(raw["test"], tokenizer, args.max_length)

    train_sampler = DistributedSampler(train_data, shuffle=True)
    val_sampler = DistributedSampler(val_data, shuffle=False)

    train_loader = DataLoader(
        train_data, batch_size=args.per_device_train_batch_size,
        sampler=train_sampler, num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    val_loader = DataLoader(
        val_data, batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler, num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # ── Model ────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    model = convert_to_moe(model, moe_config)
    model.gradient_checkpointing_enable()

    # Freeze non-MoE layers — only router + experts are trained in Stage 1
    freeze_non_moe_layers(model, moe_config)
    model.train()

    # ── DeepSpeed ────────────────────────────────────────────────────
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

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

    # ── Baseline eval ────────────────────────────────────────────────
    val_loss = evaluate(model_engine, val_loader, tokenizer)
    if rank == 0:
        print(f"Step 0 | Baseline Validation Loss: {val_loss:.4f}  "
              f"Perplexity: {math.exp(val_loss):.2f}")

    # ── Training loop ────────────────────────────────────────────────
    global_step = resume_step

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        n_skip = resume_step % len(train_loader) if epoch == start_epoch else 0
        skipped = 0

        for batch in train_loader:
            if skipped < n_skip:
                skipped += 1
                continue

            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            ce_loss = outputs.loss

            # MoE aux losses with gradients — trains the router to balance
            moe_aux = collect_aux_losses(model_engine.module, grad=True)
            aux_loss = (args.balance_alpha * moe_aux["balance_loss"]
                        + args.z_loss_beta * moe_aux["z_loss"])
            loss = ce_loss + aux_loss

            model_engine.backward(loss)
            model_engine.step()
            del outputs

            if scheduler is not None and model_engine.is_gradient_accumulation_boundary():
                scheduler.step()

            global_step += 1

            # ── Logging ─────────────────────────────────────────
            if rank == 0 and global_step % args.logging_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/ce_loss", ce_loss.item(), global_step)
                writer.add_scalar("moe/balance_loss",
                                  moe_aux["balance_loss"].item(), global_step)
                writer.add_scalar("moe/z_loss",
                                  moe_aux["z_loss"].item(), global_step)
                writer.add_scalar("moe/dropped_frac",
                                  moe_aux["dropped_frac"].item(), global_step)
                print(
                    f"[Epoch {epoch + 1}] Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"CE: {ce_loss.item():.4f} | "
                    f"Bal: {moe_aux['balance_loss'].item():.4f} | "
                    f"Z: {moe_aux['z_loss'].item():.4f}"
                )

            # ── Eval ────────────────────────────────────────────
            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model_engine, val_loader, tokenizer)
                if rank == 0:
                    writer.add_scalar("val/loss", val_loss, global_step)
                    print(f"[Eval]  Step {global_step} | "
                          f"Val Loss: {val_loss:.4f}  "
                          f"Perplexity: {math.exp(val_loss):.2f}")

            # ── Save ────────────────────────────────────────────
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                model_engine.save_checkpoint(
                    args.output_dir, tag=f"step_{global_step}",
                    client_state={"global_step": global_step},
                )

        # End of epoch — save
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        model_engine.save_checkpoint(
            args.output_dir, tag=f"epoch_{epoch + 1}",
            client_state={"global_step": global_step},
        )

    model_engine.save_checkpoint(args.output_dir, tag="final",
                                 client_state={"global_step": global_step})
    if rank == 0:
        writer.close()
        print("Stage 1 SFT complete.")


if __name__ == "__main__":
    train()

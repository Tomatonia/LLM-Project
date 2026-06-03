"""
Evaluate a trained MoE model on GSM8K and MMLU.

Handles the MoE architecture: loads the dense base model, applies MoE
conversion from ``moe_config.json``, then loads the trained DeepSpeed
or HF-format checkpoint weights.

Usage
-----
    # Evaluate a MoE training checkpoint
    python moe/eval_moe.py \
        --model_path moe/moe_qwen_gsm8k/final \
        --base_model moe/qwen_moe_init \
        --benchmark gsm8k mmlu

    # Quick sanity check
    python moe/eval_moe.py \
        --model_path moe/qwen_moe_init \
        --benchmark gsm8k --gsm8k_max_samples 50
"""

import argparse
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe.moe_layer import convert_to_moe
from eval.evaluate import evaluate_gsm8k, evaluate_mmlu


def load_moe_model(model_path: str, base_model: str):
    """
    Load an MoE model for evaluation.

    *model_path* may be:
      - A freshly-converted directory (contains ``moe_config.json`` + dense weights)
      - A DeepSpeed checkpoint from training
      - An HF-format MoE checkpoint

    *base_model* supplies the dense architecture and tokenizer when
    *model_path* is a DeepSpeed checkpoint (which lacks both).
    """
    # Where to find moe_config.json
    # If model_path = moe_qwen_gsm8k/final, look in the output root
    # If model_path = qwen_moe_init, look directly there
    moe_config_path = os.path.join(model_path, "moe_config.json")
    if not os.path.isfile(moe_config_path):
        # Check one level up (DeepSpeed checkpoint inside output dir)
        parent = os.path.dirname(model_path.rstrip("/"))
        moe_config_path = os.path.join(parent, "moe_config.json")
    if not os.path.isfile(moe_config_path):
        raise FileNotFoundError(
            f"moe_config.json not found at {model_path} or {parent}. "
            f"Pass --base_model to point to the converted dense directory."
        )

    with open(moe_config_path) as f:
        moe_config = json.load(f)

    # ── Tokenizer from base_model ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dense architecture ──────────────────────────────────────
    print(f"Loading base model from {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # ── Convert to MoE ───────────────────────────────────────────────
    convert_to_moe(model, moe_config)

    # ── Load trained weights if model_path != base_model ─────────────
    if os.path.normpath(model_path) != os.path.normpath(base_model):
        print(f"Loading trained weights from {model_path} ...")
        if _is_deepspeed_checkpoint(model_path):
            _load_deepspeed_weights(model, model_path)
        else:
            state_dict = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).state_dict()
            # Filter: only load MoE-relevant keys
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("  Model ready.")
    return model, tokenizer, moe_config


def _is_deepspeed_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        if "zero_pp_rank" in f and f.endswith("_optim_states.pt"):
            return True
    return False


def _load_deepspeed_weights(model, ckpt_path: str):
    """Load weights from a DeepSpeed ZeRO checkpoint into *model*."""
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    ckpt_dir, tag = ckpt_path, None
    if not os.path.isfile(os.path.join(ckpt_path, "latest")):
        parent = os.path.dirname(ckpt_path)
        basename = os.path.basename(ckpt_path)
        latest_file = os.path.join(parent, "latest")
        if os.path.isfile(latest_file):
            with open(latest_file) as f:
                if f.read().strip() == basename:
                    ckpt_dir, tag = parent, basename

    print(f"  Converting DeepSpeed checkpoint: dir={ckpt_dir}, tag={tag or '(latest)'}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    print("  Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MoE model")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to MoE checkpoint or converted model directory")
    p.add_argument("--base_model", type=str, default=None,
                   help="Base dense model for architecture/tokenizer "
                        "(default: same as --model_path)")
    p.add_argument("--benchmark", type=str, nargs="+",
                   default=["gsm8k"], choices=["gsm8k", "mmlu"])
    p.add_argument("--gsm8k_max_samples", type=int, default=None)
    p.add_argument("--gsm8k_batch_size", type=int, default=8)
    p.add_argument("--mmlu_max_per_subject", type=int, default=None)
    p.add_argument("--mmlu_fewshot", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--output_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_model = args.base_model or args.model_path

    model, tokenizer, moe_config = load_moe_model(args.model_path, base_model)

    results = {}
    if "gsm8k" in args.benchmark:
        print("\n" + "=" * 60)
        print(f"GSM8K  (0-shot, {moe_config['num_experts']} experts, k={moe_config['k']})")
        print("=" * 60)
        results["gsm8k_accuracy"] = evaluate_gsm8k(
            model, tokenizer,
            max_samples=args.gsm8k_max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gsm8k_batch_size,
        )

    if "mmlu" in args.benchmark:
        print("\n" + "=" * 60)
        print(f"MMLU  ({args.mmlu_fewshot}-shot)")
        print("=" * 60)
        overall, per_subject = evaluate_mmlu(
            model, tokenizer,
            num_fewshot=args.mmlu_fewshot,
            max_samples_per_subject=args.mmlu_max_per_subject,
        )
        results["mmlu_accuracy"] = overall
        results["mmlu_per_subject"] = per_subject

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in results.items():
        if k == "mmlu_per_subject":
            continue
        print(f"  {k}: {v:.4f}")

    if args.output_file:
        from datetime import datetime, timezone
        output = {
            "model_path": args.model_path,
            "base_model": base_model,
            "moe_config": moe_config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()

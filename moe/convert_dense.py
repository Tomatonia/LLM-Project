"""
Prepare a dense Qwen2.5 model for sparse MoE training.

Saves the **dense** model + a ``moe_config.json`` that describes which layers
to replace and how.  The actual conversion happens at training time so that
``AutoModelForCausalLM.from_pretrained`` loads a valid checkpoint.

Usage
-----
    # Shared expert + routed experts (recommended)
    python -m moe.convert_dense \
        --model_path Qwen/Qwen2.5-1.5B \
        --output moe/qwen_moe_init \
        --num_experts 3 --k 2 \
        --shared_intermediate_size 4096 \
        --routed_intermediate_size 3072

    # Routed-only (original)
    python -m moe.convert_dense \
        --model_path Qwen/Qwen2.5-1.5B \
        --output moe/qwen_moe_init \
        --num_experts 4 --k 2 \
        --expert_intermediate_size 4096
"""

import argparse
import json
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Prepare dense Qwen2.5 for MoE training")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output", type=str, default="moe/qwen_moe_init")
    p.add_argument("--num_experts", type=int, default=3,
                   help="Number of routed experts (default: 3)")
    p.add_argument("--k", type=int, default=2,
                   help="Experts activated per token (default: 2)")
    p.add_argument("--capacity_factor", type=float, default=2.0,
                   help="Token capacity multiplier (default: 2.0)")
    p.add_argument("--shared_intermediate_size", type=int, default=4096,
                   help="Intermediate size of the always-active shared expert. "
                        "When set, --routed_intermediate_size is also required.")
    p.add_argument("--routed_intermediate_size", type=int, default=3072,
                   help="Intermediate size per routed expert (default: 3072)")
    p.add_argument("--expert_intermediate_size", type=int, default=None,
                   help="FFN intermediate size per expert for routed-only mode "
                        "(no shared expert).  Use --shared_intermediate_size "
                        "for shared+MoE mode.")
    p.add_argument("--moe_layers", type=int, nargs="+",
                   default=list(range(1, 28, 2)),
                   help="Layer indices to replace with MoE (0-indexed). "
                        "Default: every other layer starting from 1.")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="Gaussian noise std for expert upcycling (default: 0.0)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading dense model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    print(f"  Model: {config.model_type}, {config.num_hidden_layers} layers, "
          f"d={config.hidden_size}, ffn={config.intermediate_size}")

    # Save the dense model as-is
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, max_shard_size="2GB",
                          safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    # Save the MoE conversion recipe
    moe_config = {
        "num_experts": args.num_experts,
        "k": args.k,
        "capacity_factor": args.capacity_factor,
        "moe_layers": args.moe_layers,
        "noise_std": args.noise_std,
        "hidden_size": config.hidden_size,
    }

    if args.shared_intermediate_size:
        moe_config["shared_intermediate_size"] = args.shared_intermediate_size
        moe_config["routed_intermediate_size"] = args.routed_intermediate_size
    else:
        isize = args.expert_intermediate_size or 4096
        moe_config["intermediate_size"] = isize

    with open(os.path.join(args.output, "moe_config.json"), "w") as f:
        json.dump(moe_config, f, indent=2)

    kind = "shared+MoE" if args.shared_intermediate_size else "MoE"
    print(f"  {kind}: {args.num_experts} routed experts, k={args.k}")
    print(f"Saved dense model + moe_config.json to {args.output}")


if __name__ == "__main__":
    main()

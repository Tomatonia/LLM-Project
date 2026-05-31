"""
Plot GRPO training metrics (reward and test accuracy) from TensorBoard logs.

Usage
-----
    python plot_metrics.py
    python plot_metrics.py --logdir ./grpo_qwen_gsm8k/logs --smooth 0.6
    python plot_metrics.py --tags train/loss train/mean_reward train/kl
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tags: list[str]) -> dict[str, tuple[list[int], list[float]]]:
    acc = EventAccumulator(logdir)
    acc.Reload()
    result = {}
    for tag in tags:
        if tag not in acc.Tags().get("scalars", {}):
            print(f"Warning: tag '{tag}' not found in {logdir}")
            continue
        events = acc.Scalars(tag)
        result[tag] = ([e.step for e in events], [e.value for e in events])
    return result


def exponential_smooth(values: list[float], alpha: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    smoothed = np.empty_like(v)
    smoothed[0] = v[0]
    for i in range(1, len(v)):
        smoothed[i] = alpha * v[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_metrics(
    data: dict,
    output_path: str = "grpo_metrics.png",
    smooth: float = 0.6,
    title: str = "",
):
    """Plot reward (left axis) and accuracy (right axis) on the same figure."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    colors = {
        "train/mean_reward": "#1f77b4",
        "train/loss": "#ff7f0e",
        "train/kl": "#2ca02c",
        "train/clip_frac": "#9467bd",
        "eval/gsm8k_accuracy": "#d62728",
    }
    labels = {
        "train/mean_reward": "Mean Reward",
        "train/loss": "GRPO Loss",
        "train/kl": "KL Divergence",
        "train/clip_frac": "Clip Fraction",
        "eval/gsm8k_accuracy": "GSM8K Accuracy",
    }

    for tag, (steps, values) in data.items():
        # Don't smooth eval metrics — they only log every ~100 steps and
        # EMA would badly distort sparse, step-like data.
        is_eval = tag.startswith("eval/")
        if smooth > 0 and not is_eval:
            y = exponential_smooth(values, smooth)
        else:
            y = np.array(values)
        label = labels.get(tag, tag)
        ax1.plot(steps, y, label=label, color=colors.get(tag),
                 linewidth=1.2, alpha=0.9, marker="o" if is_eval else "")

    ax1.set_xlabel("Step")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    if title:
        ax1.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot GRPO training metrics")
    p.add_argument("--logdir", type=str, default="./grpo_qwen_gsm8k/logs")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--smooth", type=float, default=0.6)
    p.add_argument("--tags", type=str, nargs="+",
                   default=["train/mean_reward", "eval/gsm8k_accuracy"])
    p.add_argument("--title", type=str, default="GRPO Training on GSM8K")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.logdir.rstrip("/")), "grpo_metrics.png"
        )
    data = load_scalars(args.logdir, args.tags)
    if not data:
        print(f"No matching tags found in {args.logdir}")
        return
    plot_metrics(data, args.output, smooth=args.smooth, title=args.title)


if __name__ == "__main__":
    main()

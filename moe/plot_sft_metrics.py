"""
Plot MoE SFT training metrics (train loss and validation loss) from TensorBoard logs.

Usage
-----
    python moe/plot_sft_metrics.py
    python moe/plot_sft_metrics.py --smooth 0.6
    python moe/plot_sft_metrics.py --tags train/loss train/ce_loss val/loss
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
        seen: dict[int, float] = {}
        for e in events:
            seen[e.step] = e.value
        steps = sorted(seen.keys())
        result[tag] = (steps, [seen[s] for s in steps])
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
    output_path: str = "sft_metrics.png",
    smooth: float = 0.6,
    title: str = "",
):
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        "train/loss": "#1f77b4",
        "train/ce_loss": "#ff7f0e",
        "val/loss": "#d62728",
        "moe/balance_loss": "#8c564b",
        "moe/z_loss": "#e377c2",
        "moe/dropped_frac": "#7f7f7f",
    }
    labels = {
        "train/loss": "Train Loss",
        "train/ce_loss": "Train CE Loss",
        "val/loss": "Validation Loss",
        "moe/balance_loss": "MoE Balance Loss",
        "moe/z_loss": "MoE Z-Loss",
        "moe/dropped_frac": "MoE Dropped Fraction",
    }

    for tag, (steps, values) in data.items():
        is_eval = tag.startswith("val/")
        if smooth > 0 and not is_eval:
            y = exponential_smooth(values, smooth)
        else:
            y = np.array(values)
        label = labels.get(tag, tag)
        ax.plot(steps, y, label=label, color=colors.get(tag),
                linewidth=1.2, alpha=0.9, marker="o" if is_eval else "")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot MoE SFT training metrics")
    p.add_argument("--logdir", type=str, default="moe/moe_sft_gsm8k/logs")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--smooth", type=float, default=0.6)
    p.add_argument("--tags", type=str, nargs="+",
                   default=["train/loss", "val/loss"])
    p.add_argument("--title", type=str, default="MoE SFT Training on GSM8K")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.logdir.rstrip("/")), "sft_metrics.png"
        )
    data = load_scalars(args.logdir, args.tags)
    if not data:
        print(f"No matching tags found in {args.logdir}")
        return
    plot_metrics(data, args.output, smooth=args.smooth, title=args.title)


if __name__ == "__main__":
    main()

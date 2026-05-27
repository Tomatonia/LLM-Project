"""
Extract train/val loss curves from TensorBoard logs and save as a PNG.

Usage
-----
    python sft/plot_loss.py --logdir ./sft_qwen_gsm8k/logs
    python sft/plot_loss.py --logdir ./sft_qwen_gsm8k/logs --smooth 0.6  # more smoothing
    python sft/plot_loss.py --logdir ./grpo_qwen_gsm8k/logs --tag train/mean_reward --ylabel "Mean Reward"
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tags: list[str]) -> dict[str, tuple[list[int], list[float]]]:
    """Load scalar time series from a TensorBoard log directory."""
    acc = EventAccumulator(logdir)
    acc.Reload()

    result = {}
    for tag in tags:
        if tag not in acc.Tags().get("scalars", {}):
            print(f"Warning: tag '{tag}' not found in {logdir}")
            continue
        events = acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        result[tag] = (steps, values)
    return result


def exponential_smooth(values: list[float], alpha: float) -> np.ndarray:
    """Exponential moving average: ema_t = alpha * x_t + (1-alpha) * ema_{t-1}."""
    v = np.array(values, dtype=np.float64)
    smoothed = np.empty_like(v)
    smoothed[0] = v[0]
    for i in range(1, len(v)):
        smoothed[i] = alpha * v[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def plot_losses(
    data: dict,
    output_path: str = "loss_curve.png",
    smooth: float = 0.0,
    ylabel: str = "Loss",
    title: str = "",
    figsize: tuple = (10, 5),
):
    """Plot loss curves and save to file."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = {"train": "#1f77b4", "val": "#d62728", "eval": "#d62728"}
    linestyles = {"train": "-", "val": "--", "eval": "--"}

    for tag, (steps, values) in data.items():
        # Infer label from tag (e.g. "train/loss" → "train")
        label = tag.split("/")[-1]
        series = tag.split("/")[0]

        color = colors.get(series, None)
        ls = linestyles.get(series, "-")

        y = np.array(values)
        if smooth > 0:
            y = exponential_smooth(values, smooth)

        ax.plot(steps, y, label=f"{series} {label}", color=color, linestyle=ls, linewidth=1.2, alpha=0.9)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, required=True, help="Path to TensorBoard log directory")
    p.add_argument("--output", type=str, default=None, help="Output PNG path (default: <logdir>/../loss_curve.png)")
    p.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing factor (0 = none, 0.6 = moderate)")
    p.add_argument("--tags", type=str, nargs="+", default=None,
                   help="Specific tags to plot (default: train/loss val/loss)")
    p.add_argument("--ylabel", type=str, default="Loss")
    p.add_argument("--title", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()

    if args.tags is None:
        tags = ["train/loss", "val/loss"]
    else:
        tags = args.tags

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.logdir.rstrip("/")), "loss_curve.png")

    data = load_scalars(args.logdir, tags)
    if not data:
        print(f"No matching tags found in {args.logdir}")
        return

    plot_losses(data, args.output, smooth=args.smooth, ylabel=args.ylabel, title=args.title)


if __name__ == "__main__":
    main()

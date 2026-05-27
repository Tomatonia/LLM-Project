# LLM-Project

Final project for the LLM Applications course (2026 spring semester). Fine-tuning language models on mathematical reasoning tasks.

## Structure

```
sft/
├── train_sft.py          # Supervised fine-tuning with DeepSpeed
├── ds_config.json        # DeepSpeed ZeRO Stage 2 configuration
└── plot_loss.py          # Export loss curves from TensorBoard logs

grpo/
├── train_grpo.py         # GRPO training script
├── rewards.py            # GSM8K reward functions
├── ds_config_grpo.json   # DeepSpeed configuration
└── ARCHITECTURE.md       # GRPO pipeline documentation

eval/
└── evaluate.py           # GSM8K + MMLU evaluation (standalone)
```

## Setup

Install dependencies:

```bash
pip install deepspeed torch transformers datasets tensorboard matplotlib
```

Flash-Attention 2 is also required. Install a pre-built wheel matching your configuration
(CUDA version, torch version, Python version, CXX11 ABI), since compiling from source
takes a significant amount of time.

For example, the following installs for CUDA 12, torch 2.7, Python 3.10, and CXX11 ABI false:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**Note:** The training scripts set `HF_ENDPOINT=https://hf-mirror.com` for downloading models
and datasets behind the GFW. Remove or change this if you are outside mainland China.

## SFT Training

```bash
cd sft
```

Supervised fine-tunes `Qwen/Qwen2.5-1.5B` on GSM8K math problems from the
[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset.

The dataset is split 90/10 into train (6,610) and validation (735) sets. With 2 GPUs
and the default settings below, one epoch is ~1,653 iterations and one training run
(2 epochs) is ~3,306 iterations (~1,653 optimizer steps after gradient accumulation).

```bash
deepspeed --num_gpus=2 sft/train_sft.py --deepspeed_config ds_config.json
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-1.5B` | Base model from HuggingFace |
| `--num_epochs` | `2` | Number of training epochs |
| `--learning_rate` | `2e-5` | Peak learning rate (linear warmup + decay) |
| `--warmup_steps` | `100` | LR warmup steps |
| `--per_device_train_batch_size` | `2` | Micro batch size per GPU |
| `--per_device_eval_batch_size` | `2` | Eval batch size per GPU |
| `--gradient_accumulation_steps` | `2` | Accumulate before each optimizer step |
| `--max_length` | `1024` | Max tokenized sequence length |
| `--output_dir` | `./sft_qwen_gsm8k` | Checkpoint and log directory |
| `--logging_steps` | `10` | Log train loss every N iterations |
| `--eval_steps` | `100` | Run validation every N iterations |
| `--save_steps` | `1000` | Save checkpoint every N iterations |
| `--resume_from_checkpoint` | — | Path to a checkpoint directory to resume from |

### Data format

Each example is formatted as:

```
Problem: {problem}
Solution:
{solution}
```

Labels mask the prompt tokens (through `\nSolution:\n`) with -100 so loss is computed
only on the solution tokens. A prefix-consistency check guards against tokenizer
alignment errors when the prompt is tokenized separately to find the label mask boundary.

### DeepSpeed configuration (`ds_config.json`)

ZeRO Stage 2 with BF16 and no CPU offload. ZeRO-2 shards optimizer states and gradients
across GPUs while keeping full model parameters on each device — sufficient for a 1.5B
model (~13.5 GB GPU per card) and avoids the CPU memory pressure of ZeRO-3 offload.

The scheduler is handled in code (`get_linear_schedule_with_warmup`), not in the
DeepSpeed config, to avoid double-scheduler conflicts. The scheduler is stepped only on
gradient accumulation boundaries so LR decay is correctly aligned with optimizer updates.

### Resuming from a checkpoint

```bash
deepspeed --num_gpus=2 sft/train_sft.py --resume_from_checkpoint ./sft_qwen_gsm8k/step_1000 --deepspeed_config ds_config.json
```

The resume path can point to a step directory (`step_1000`), an epoch directory
(`epoch_1`), or the `final` checkpoint. The script restores global step, optimizer
state, and LR scheduler state from the checkpoint.

### Visualizing loss curves

```bash
# Raw curves
python plot_loss.py --logdir sft_qwen_gsm8k/logs

# With EMA smoothing
python plot_loss.py --logdir sft_qwen_gsm8k/logs --smooth 0.6

# Custom output
python plot_loss.py --logdir sft_qwen_gsm8k/logs --output loss.png --title "SFT on GSM8K"
```

The script reads TensorBoard event files directly and supports any scalar tags
(e.g., `train/loss`, `val/loss`, `train/mean_reward` from GRPO).

## Logging

TensorBoard logs are written to `--output_dir/logs`. Launch with:

```bash
tensorboard --logdir sft_qwen_gsm8k/logs
```

## Evaluation

Evaluate a trained model on GSM8K (0-shot, official test set) and MMLU (5-shot).
The evaluation script is standalone and works with any HF-compatible model path.
`--model_path` defaults to `Qwen/Qwen2.5-1.5B` (the base model).

```bash
# Base model, GSM8K only
python ../val/evaluate.py

# Trained checkpoint, both benchmarks
python ../eval/evaluate.py --model_path ./sft_qwen_gsm8k/final --benchmark gsm8k mmlu

# Quick check (limited samples)
python eval/evaluate.py --model_path ./sft_qwen_gsm8k/final \
    --benchmark gsm8k mmlu --gsm8k_max_samples 100 --mmlu_max_per_subject 20

# MMLU only, specific subjects, save results
python eval/evaluate.py --model_path ./sft_qwen_gsm8k/final \
    --benchmark mmlu --mmlu_subjects high_school_mathematics college_mathematics \
    --output_file results.json
```

### Benchmarks

| Benchmark | Few-shot | Dataset | Metric |
|-----------|----------|---------|--------|
| GSM8K | 0-shot | `gsm8k` test set (1,319 examples) | Exact match after `####` |
| MMLU | 5-shot | `cais/mmlu` test set (14,042 examples, 57 subjects) | Label logprob (matches lm_eval `acc`) |

For GSM8K the script uses the official `gsm8k` test split (not the training val split)
to avoid data leakage. For MMLU it uses 5-shot prompting with examples from the `dev`
split, and scores each choice by the log-probability of its letter label (A/B/C/D)
as the next token — equivalent to `lm_eval`'s default MMLU metric.

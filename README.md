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
├── ds_config_grpo.json   # DeepSpeed ZeRO-2 configuration
└── plot_metrics.py       # Plot reward and accuracy curves

moe/
├── expert.py             # Expert FFN (Qwen2MLP architecture)
├── router.py             # TopKRouter with temperature-scaled softmax
├── moe_layer.py          # MoEBlockWithShared / MoEBlock / convert_to_moe
├── convert_dense.py      # Prepare dense model + moe_config.json
├── train_moe.py          # GRPO + MoE training script
├── diagnostic.py         # Init correctness check (compare vs dense)
└── ds_config_moe.json    # ZeRO-2, BF16, grad_accum=1

eval/
├── eval_moe.py           # GSM8K + MMLU evaluation for MoE checkpoints
└── evaluate.py           # GSM8K + MMLU evaluation for SFT and GRPO checkpoints
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

## SFT Training

Supervised fine-tunes `Qwen/Qwen2.5-1.5B` on GSM8K math problems from the
[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset.

The dataset is split 90/10 into train (6,610) and validation (735) sets. With 2 GPUs
and the default settings below, one epoch is ~1,653 iterations and one training run
(2 epochs) is ~3,306 iterations (~1,653 optimizer steps after gradient accumulation).

```bash
deepspeed --num_gpus=2 sft/train_sft.py --deepspeed_config sft/ds_config.json
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
| `--output_dir` | `sft/sft_qwen_gsm8k` | Checkpoint and log directory |
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
deepspeed --num_gpus=2 sft/train_sft.py --resume_from_checkpoint sft/sft_qwen_gsm8k/step_1000 --deepspeed_config ds_config.json
```

The resume path can point to a step directory (`step_1000`), an epoch directory
(`epoch_1`), or the `final` checkpoint. The script restores global step, optimizer
state, and LR scheduler state from the checkpoint.

### Visualizing loss curves

```bash
# Raw curves
python sft/plot_loss.py --logdir sft/sft_qwen_gsm8k/logs

# With EMA smoothing
python sft/plot_loss.py --logdir sft/sft_qwen_gsm8k/logs --smooth 0.6

# Custom output
python sft/plot_loss.py --logdir sft/sft_qwen_gsm8k/logs --output loss.png --title "SFT on GSM8K"
```

The script reads TensorBoard event files directly and supports any scalar tags
(e.g., `train/loss`, `val/loss`, `train/mean_reward` from GRPO).

## GRPO Training

Group Relative Policy Optimization (GRPO) is a reinforcement learning stage that runs
after SFT (or from the base model) to further improve mathematical reasoning. Unlike PPO,
GRPO eliminates the critic network by normalising rewards *within a group* of G responses
to the same prompt. The policy is updated with a clipped surrogate objective plus a KL
penalty against a frozen reference model.

GRPO trains from the base model by default (`Qwen/Qwen2.5-1.5B`). To train on top of an
SFT checkpoint, pass `--model_path sft/sft_qwen_gsm8k/final`.

The training uses all 7,345 GSM8K prompts from NuminaMath-CoT (no train/val split needed
since the model generates its own rollouts — no ground-truth leakage). With 2 GPUs and the
default settings, one epoch is ~919 iterations.

```bash
deepspeed --num_gpus=2 grpo/train_grpo.py --deepspeed_config grpo/ds_config_grpo.json
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen2.5-1.5B` | Starting checkpoint (base model or SFT) |
| `--group_size` | `4` | G: responses sampled per prompt |
| `--kl_coef` | `0.04` | Weight of KL penalty against reference model |
| `--clip_epsilon` | `0.2` | PPO clipping threshold |
| `--num_epochs` | `2` | Number of training epochs |
| `--learning_rate` | `1e-6` | Peak learning rate (lower than SFT for stable RL) |
| `--warmup_steps` | `20` | LR warmup steps |
| `--per_device_prompt_batch_size` | `4` | Prompts per GPU per step |
| `--max_new_tokens` | `256` | Max tokens for rollout generation |
| `--temperature` | `1.0` | Sampling temperature for rollouts |
| `--top_p` | `0.9` | Nucleus sampling threshold |
| `--output_dir` | `grpo/grpo_qwen_gsm8k` | Checkpoint and log directory |
| `--logging_steps` | `5` | Log train metrics every N iterations |
| `--eval_steps` | `100` | Run evaluation on official GSM8K test set |
| `--save_steps` | `500` | Save checkpoint every N iterations |

### How it works

Each training step:

1. **Generate rollouts** — B prompts × G=4 responses each = B×G candidate solutions
   generated in a single batched call (sampling with temperature 1.0).
2. **Compute rewards** — extract answer from `####` or `\boxed{}`, compare to
   ground truth (binary 0/1).
3. **Group-relative advantages** — A_i = (r_i − mean) / (std + ε) normalised per prompt.
4. **Forward passes** — compute token-level logprobs for old policy (no grad),
   reference model (no grad), and new policy (with grad).
5. **GRPO loss** — clipped surrogate + KL penalty, applied to response tokens only.
6. **Backward + step** via DeepSpeed ZeRO-2.

### Evaluation during training

Every `--eval_steps` iterations, the script evaluates accuracy on the official
`openai/gsm8k` test set (1,319 examples) using batched greedy decoding. This is
the same metric used for final evaluation.

### Reward function

The reward is binary: 1.0 if the final answer matches the ground truth after
normalisation (number parsing, fraction handling, trailing zero stripping), 0.0
otherwise. Both `####` (GSM8K convention) and `\boxed{}` (LaTeX convention) are
recognised as answer markers.

### DeepSpeed configuration (`ds_config_grpo.json`)

ZeRO Stage 2 with BF16 and no CPU offload (same rationale as SFT). The scheduler is
handled in code to avoid double-scheduler conflicts. Gradient checkpointing is enabled
on the policy model; rollout generation temporarily switches to eval mode for fast
KV-cache inference.

### Visualizing metrics

```bash
# Reward + accuracy (default)
python grpo/plot_metrics.py

# With specific tags
python grpo/plot_metrics.py --tags train/mean_reward eval/gsm8k_accuracy train/loss train/kl

# No smoothing
python grpo/plot_metrics.py --smooth 0
```

## MoE Training

Mixture-of-Experts GRPO training replaces half the dense FFN layers in
`Qwen/Qwen2.5-1.5B` with shared+routed MoE blocks (1 shared expert + 3 routed,
top-2 gating).  Custom dispatch/combine with partitioned weight initialization
for proper ZeRO-2 integration.

```bash
# 1. Convert the dense model (saves dense weights + moe_config.json)
python -m moe.convert_dense \
    --model_path Qwen/Qwen2.5-1.5B \
    --output moe/qwen_moe_init

# 2. Train MoE with GRPO
deepspeed --num_gpus=2 moe/train_moe.py \
    --model_path moe/qwen_moe_init \
    --output_dir moe/moe_qwen_gsm8k \
    --deepspeed_config moe/ds_config_moe.json

# 3. Evaluate
python moe/eval_moe.py \
    --model_path moe/moe_qwen_gsm8k/final \
    --base_model moe/qwen_moe_init \
    --benchmark gsm8k mmlu
```

### Architecture

14 of 28 transformer layers (every other, starting from layer 1) have their
`Qwen2MLP` replaced with an `MoEBlockWithShared`: one shared expert (4096 dims,
always active) + 3 routed experts (3072 dims each, top-2 gating).  The shared
expert covers dense-FFN dims [0, 4096) at full strength on every token; the 3
routed experts partition the remaining dims with staggered overlapping slices.

Total: ~1.83B params, fits 2×24 GB GPUs under ZeRO-2 (~21 GB peak at batch
size 4).  For lower memory, reduce `--per_device_prompt_batch_size`.

### DeepSpeed configuration

ZeRO Stage 2, BF16, no CPU offload, `gradient_accumulation_steps: 1`.
ZeRO-2 keeps full params per GPU — no `GatheredParameters` or weight
syncing needed.  Generation, forward passes, and backward all work directly.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `moe/qwen_moe_init` | Converted model directory |
| `--per_device_prompt_batch_size` | `4` | Prompts per GPU per step |
| `--temperature` | `1.0` | Rollout sampling temperature |
| `--eval_batch_size` | `16` | GSM8K eval batch (no-grad, greedy) |
| `--balance_alpha` | `0.01` | Load balancing loss weight |
| `--z_loss_beta` | `0.001` | Router Z-loss weight |
| `--num_epochs` | `1` | 3736 steps / `per_device_prompt_batch_size` |

### How it works

Same GRPO algorithm as dense training:

1. **Generate rollouts** — each rank independently generates G=4 rollouts
   per prompt using `model_engine.module.generate()` (ZeRO-2, full params).
2. **Reward** — early training uses format bonus (0.1 for `####`, 1.0 for
   correct) to bootstrap the delimiter; revert to binary after format stabilizes.
3. **Forward passes** — old policy (no grad), reference model (no grad,
   offloaded to CPU between uses), new policy (with grad, gradient checkpointed).
4. **GRPO loss + MoE aux loss** — `balance_alpha × L_balance` added.
5. **Backward + step** via DeepSpeed ZeRO-2.

### Router design

`TopKRouter` with temperature-scaled softmax: `softmax(logits / 2.0) × k`.
Weights sum to k (≈2.0), so each selected expert contributes at full strength
(~1.0×) rather than half (~0.5×), matching the dense FFN output magnitude at
initialization.

## Evaluation

Evaluate a trained model on GSM8K (0-shot, official test set) and MMLU (5-shot).
The evaluation script is standalone and works with any HF-compatible model path.
`--model_path` defaults to `Qwen/Qwen2.5-1.5B` (the base model).

```bash
# Base model, GSM8K only
python eval/evaluate.py

# Trained checkpoint, both benchmarks
python eval/evaluate.py --model_path sft/sft_qwen_gsm8k/final --benchmark gsm8k mmlu

# Quick check (limited samples)
python eval/evaluate.py --model_path sft/sft_qwen_gsm8k/final \
    --benchmark gsm8k mmlu --gsm8k_max_samples 100 --mmlu_max_per_subject 20

# MMLU only, specific subjects, save results
python eval/evaluate.py --model_path sft/sft_qwen_gsm8k/final \
    --benchmark mmlu --mmlu_subjects high_school_mathematics college_mathematics \
    --output_file eval/sft_mmlu_results.json

# MoE model, GSM8K and MMLU tasks
python eval/eval_moe.py --model_path moe/moe_qwen_gsm8k/final \
    --base_model moe/qwen_moe_init \
    --benchmark gsm8k mmlu \
    --output_file eval/moe_results.json
```

### Benchmarks

| Benchmark | Few-shot | Dataset | Metric |
|-----------|----------|---------|--------|
| GSM8K | 0-shot | `openai/gsm8k` test set (1,319 examples) | Exact match after `####` |
| MMLU | 5-shot | `cais/mmlu` test set (14,042 examples, 57 subjects) | Label logprob (matches lm_eval `acc`) |

For GSM8K the script uses the official `gsm8k` test split (not the training val split)
to avoid data leakage. For MMLU it uses 5-shot prompting with examples from the `dev`
split, and scores each choice by the log-probability of its letter label (A/B/C/D)
as the next token — equivalent to `lm_eval`'s default MMLU metric.

## Logging

TensorBoard logs are written to `--output_dir/logs`. Launch with:

```bash
tensorboard --logdir sft/sft_qwen_gsm8k/logs
tensorboard --logdir grpo/grpo_qwen_gsm8k/logs
tensorboard --logdir moe/moe_qwen_gsm8k/logs
```
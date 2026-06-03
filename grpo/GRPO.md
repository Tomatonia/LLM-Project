# GRPO Pipeline Architecture

## Overview

Group Relative Policy Optimization (GRPO) is a reinforcement learning method
introduced by DeepSeek for fine-tuning LLMs. Unlike PPO, GRPO eliminates the
critic (value) network by normalizing rewards *within a group* of responses to
the same prompt. This halves memory consumption and simplifies training.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP (per step)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GSM8K Train Set                                                    │
│  ┌──────────────────┐                                               │
│  │ B prompts        │─── DistributedSampler ──→ B prompts per GPU   │
│  │ (problem text)    │                                               │
│  └──────────────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 1. GENERATE ROLLOUTS                         │                   │
│  │    For each prompt, sample G=4 responses     │                   │
│  │    model.generate(do_sample=True, T=1.0)     │                   │
│  │    Result: B×G (prompt+response) pairs       │                   │
│  └──────────────────────────────────────────────┘                   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 2. COMPUTE REWARDS                           │                   │
│  │    Extract answer after "####"               │                   │
│  │    Compare with ground truth (exact match)   │                   │
│  │    r_i ∈ {0, 1}                              │                   │
│  └──────────────────────────────────────────────┘                   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 3. GROUP-RELATIVE ADVANTAGES                 │                   │
│  │    Per prompt (group of G responses):        │                   │
│  │    A_i = (r_i - mean(r_group)) /             │                   │
│  │          (std(r_group) + 1e-8)               │                   │
│  │    Normalised independently per prompt.      │                   │
│  └──────────────────────────────────────────────┘                   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 4. FORWARD PASSES                            │                   │
│  │                                              │                   │
│  │  ┌──────────────────────┐                    │                   │
│  │  │ Policy Model (θ)     │ → new_logprobs    │  GRAD ENABLED     │
│  │  └──────────────────────┘                    │                   │
│  │  ┌──────────────────────┐                    │                   │
│  │  │ Policy Model (θ)     │ → old_logprobs    │  torch.no_grad()  │
│  │  └──────────────────────┘                    │                   │
│  │  ┌──────────────────────┐                    │                   │
│  │  │ Reference Model (θ₀) │ → ref_logprobs    │  torch.no_grad()  │
│  │  │ (frozen initial copy) │                    │                   │
│  │  └──────────────────────┘                    │                   │
│  │                                              │                   │
│  │  All forward on concatenated [prompt+resp]   │                   │
│  │  Token-level logprobs extracted.             │                   │
│  └──────────────────────────────────────────────┘                   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 5. GRPO LOSS                                 │                   │
│  │                                              │                   │
│  │  ratio_t = exp(new_logprob_t - old_logprob_t)│                   │
│  │                                              │                   │
│  │  L_policy = -mean(                           │                   │
│  │    min(ratio_t × A_i,                        │                   │
│  │        clip(ratio_t, 0.8, 1.2) × A_i)       │                   │
│  │  )                                           │                   │
│  │                                              │                   │
│  │  L_KL = mean(new_logprob_t - ref_logprob_t)  │                   │
│  │                                              │                   │
│  │  L = L_policy + 0.04 × L_KL                  │                   │
│  │                                              │                   │
│  │  Loss computed only on RESPONSE tokens.      │                   │
│  │  Prompt tokens and padding are masked.       │                   │
│  └──────────────────────────────────────────────┘                   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 6. BACKWARD + STEP (DeepSpeed ZeRO-2)        │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Files

```
grpo/
├── train_grpo.py          # Main GRPO training script
└── ds_config_grpo.json    # DeepSpeed ZeRO-2 configuration

eval/
└── evaluate.py            # GSM8K + MMLU evaluation (standalone)
```

## Key Design Decisions

### Why ZeRO-2 instead of ZeRO-3?
GRPO requires generation during training (`model.generate()`). With ZeRO-3,
parameters are sharded across GPUs, which complicates generation (params must
be gathered first). ZeRO-2 keeps full parameters on each GPU while sharding
optimizer states and gradients — sufficient for a 1.5B model (~3 GB params).

### Why a separate reference model?
The KL penalty prevents the policy from diverging too far from the initial
model. The reference model is a frozen copy of the starting checkpoint. Forward
passes through it are done under `torch.no_grad()` and do not consume optimizer
memory.

### Group-relative advantage
Normalizing per-prompt (not globally) ensures that advantages reflect relative
response quality. If all G responses to a prompt are wrong, they all get
negative or zero advantage — the policy isn't rewarded for mediocre outputs.

### Token-level loss with per-response advantage
The scalar advantage A_i is broadcast to every token in response i. Each token
contributes independently to the policy gradient, but all tokens in a response
share the same advantage sign.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| group_size (G) | 4 | Responses per prompt |
| clip_epsilon | 0.2 | PPO clipping threshold |
| kl_coef (β) | 0.08 | KL penalty weight |
| temperature | 1.0 | Sampling temperature for rollouts |
| top_p | 0.9 | Nucleus sampling threshold |
| learning_rate | 1e-6 | Much lower than SFT (2e-5) for stable RL |
| per_device_prompt_batch_size | 4 | Prompts per GPU per step |

## Evaluation

Two benchmarks are evaluated via `eval/evaluate.py`:

1. **GSM8K** (0-shot): Official `gsm8k` test set (1,319 examples). Greedy decoding,
   answer extracted after `####`, compared via exact match after normalization.
   Equivalent to lm_eval's GSM8K metric.

2. **MMLU** (5-shot): `cais/mmlu` test set (14,042 examples, 57 subjects). 5-shot
   examples from the `dev` split. Label logprob scoring (A/B/C/D) — matches
   lm_eval's `acc` metric.

## Usage

```bash
# GRPO on top of base model
deepspeed --num_gpus=2 train_grpo.py \
    --output_dir grpo/grpo_qwen_gsm8k \
    --num_epochs 2 \
    --deepspeed_config ds_config_grpo.json

# Evaluation: before (SFT) vs after (GRPO)
python eval/evaluate.py --model_path sft/sft_qwen_gsm8k/final --benchmark gsm8k mmlu
python eval/evaluate.py --model_path grpo/grpo_qwen_gsm8k/final --benchmark gsm8k mmlu
```

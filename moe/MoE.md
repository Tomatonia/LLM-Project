# Mixture-of-Experts on Qwen2.5-1.5B

## Architecture

Shared expert (4096 dims, always active) + 3 routed experts (3072 dims each,
top-2 gating).  The shared expert covers dense-FFN dims [0, 4096) at full
strength on every token, preventing hidden-state collapse.  The 3 routed
experts partition the remaining [4096, 8960) with evenly-spaced overlapping
slices so any top-2 pair covers ≥96% of the routed range.

Total: ~1.83B params (vs 1.54B dense), fits 2×24 GB GPUs under ZeRO-2.

## Implementation

### Router (`router.py`)

`TopKRouter`: learned gate `nn.Linear(d, E)`, top-k selection, capacity-based
token dropping.  Key design choices:

- **Temperature-scaled softmax × k**: `softmax(logits / 2.0) × k` so weights
  sum to k (≈2.0) instead of 1.0.  Each selected expert contributes at full
  strength (≈1.0×) rather than half (≈0.5×), matching the dense FFN magnitude.
- **Default capacity_factor = 2.0**: capacity = `ceil(N × k/E × 2.0)`, enough
  that no token is ever dropped even under maximum routing imbalance.
- **Dispatch mask built elementwise**: `dispatch_mask[idx, e, slots]` uses
  paired `idx` and `slots` tensors (not slice broadcasting), so each token
  gets exactly one capacity slot per selection.

### MoEBlock (`moe_layer.py`)

Two modes, auto-detected from `moe_config.json`:

| Config key | Block class | Structure |
|---|---|---|
| `shared_intermediate_size` present | `MoEBlockWithShared` | 1 shared + E routed |
| `intermediate_size` present | `MoEBlock` | E routed only |

`MoEBlockWithShared.forward`: `y = shared(x) + Σ w_i × routed_i(x)`.
The shared expert processes every token at weight 1.0 (no routing).
Only the routed experts use top-k gating.

### Weight Initialization (`init_from_dense`)

Shared expert copies dense dims [0, 4096).  Routed experts are staggered:

```
stride = (dense_dims - shared_dims) / num_routed
Expert i: start = shared_dims + i × stride, len = routed_dims (wrapping)
```

With 8960 total, 4096 shared, 4864 routable, 3 routed: stride = 1621.
Expert 0 covers [4096, 7168), Expert 1 covers [5717, 8789), Expert 2 wraps
to [7338, 8960) + [4096, 5506).  Any 2 of 3 cover ≥96.5% of the routed
range.  Combined with shared: **99.3%** of the original FFN at init.
Noise std defaults to 0.0 (not needed with partitioned init).

## File Plan

```
moe/
├── expert.py            # Expert FFN (Qwen2MLP architecture)
├── router.py            # TopKRouter + load_balance_loss + router_z_loss
├── moe_layer.py         # MoEBlock / MoEBlockWithShared / convert_to_moe
├── convert_dense.py     # Save dense model + moe_config.json
├── train_moe.py         # GRPO training loop
├── eval_moe.py          # Standalone GSM8K + MMLU evaluation
├── diagnostic.py        # Quick init correctness check (run on server)
├── ds_config_moe.json   # ZeRO-2, BF16, grad_accum=1
└── MoE.md               # This file
```

## Training

### DeepSpeed config

ZeRO Stage 2, BF16, no CPU offload, gradient_accumulation_steps=1.
ZeRO-2 keeps full params on each GPU — no all-gather needed during
forward/generation, so `model_engine.module.generate()` works directly
without `GatheredParameters`.

Peak GPU memory: ~21 GB per card at per_device_prompt_batch_size=4.

### Key commands

```bash
# Prepare model
python -m moe.convert_dense \
    --model_path Qwen/Qwen2.5-1.5B \
    --output moe/qwen_moe_init

# Train
deepspeed --num_gpus=2 moe/train_moe.py \
    --per_device_prompt_batch_size 4 \
    --deepspeed_config moe/ds_config_moe.json

# Resume with changed reward (after format stabilizes)
deepspeed --num_gpus=2 moe/train_moe.py \
    --resume_from_checkpoint moe/moe_qwen_gsm8k/step_400 \
    --per_device_prompt_batch_size 4 \
    --deepspeed_config moe/ds_config_moe.json

# Evaluate
python moe/eval_moe.py \
    --model_path moe/moe_qwen_gsm8k/final \
    --base_model moe/qwen_moe_init \
    --benchmark gsm8k mmlu

# Diagnostic (single-pass comparison vs dense)
python moe/diagnostic.py
```

### Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--per_device_prompt_batch_size` | 4 | Prompts per GPU per step |
| `--temperature` | 1.0 | Rollout sampling temperature |
| `--eval_batch_size` | 16 | GSM8K eval batch (no-grad, greedy) |
| `--balance_alpha` | 0.01 | MoE load-balancing loss weight |
| `--z_loss_beta` | 0.001 | Router Z-loss weight |

### Format reward bootstrapping

The base Qwen2.5 wasn't trained with `####`, so at init the model rarely
samples the delimiter and reward is always 0 (no gradient signal).  The
reward function includes a format bonus during early training:
0.0 = no `####`, 0.1 = has `####` but wrong answer, 1.0 = correct.
Once format stabilizes, revert to binary (0/1) reward and resume from
checkpoint.

This design was **not** tested in my experiments.

## Memory & Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Shared expert | 4096 dims, always active | Prevents hidden-state collapse |
| Routed experts | 3 × 3072 dims, k=2 | Fits 24 GB ZeRO-2; 2/3 pairs cover full range |
| DeepSpeed | ZeRO-2 | Required — ZeRO-3 not supported by custom MoE |
| Grad accumulation | 1 | One backward + step per batch |
| Capacity factor | 2.0 | No token dropping regardless of routing |
| Noise std | 0.0 | Partitioned init makes it unnecessary |
| Router temp | 2.0 (softmax) | Balanced weights at init; logit variance breaks topk ties |
| Weight scaling | × k (sum to k) | Full-strength expert contribution with partitioned slices |
| Gen model | None | ZeRO-2 keeps full params; generate() works directly |
| Rank sync | None | Each rank generates own rollouts independently |

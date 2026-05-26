# LLM-Project

Final project for the LLM Applications course (2026 spring semester). Fine-tuning language models on mathematical reasoning tasks.

## Structure

```
sft/
├── train_sft.py      # Supervised fine-tuning with DeepSpeed
└── ds_config.json    # DeepSpeed ZeRO Stage 2 configuration
```

## Setup

Install dependencies:

```bash
pip install deepspeed torch transformers datasets tensorboard
```

## SFT Training

Fine-tunes `Qwen/Qwen2.5-1.5B` on GSM8K problems from the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset.

```bash
deepspeed --num_gpus=2 sft/train_sft.py --deepspeed_config sft/ds_config.json
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-1.5B` | Base model from HuggingFace |
| `--num_epochs` | `3` | Number of training epochs |
| `--learning_rate` | `2e-5` | Peak learning rate |
| `--per_device_train_batch_size` | `4` | Micro batch size per GPU |
| `--max_length` | `1024` | Max tokenized sequence length |
| `--output_dir` | `./sft_qwen_gsm8k` | Checkpoint and log directory |
| `--resume_from_checkpoint` | — | Path to a checkpoint directory to resume from |

### Resuming from a checkpoint

```bash
deepspeed --num_gpus=2 sft/train_sft.py --deepspeed_config sft/ds_config.json --resume_from_checkpoint ./sft_qwen_gsm8k/step_1000
```

## Logging

TensorBoard logs are written to `--output_dir/logs`. Launch with:

```bash
tensorboard --logdir ./sft_qwen_gsm8k/logs
```

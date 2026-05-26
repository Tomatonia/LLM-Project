import argparse
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from functools import partial


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        problem = example["problem"]
        solution = example["solution"]

        # Build the full text: prompt + solution
        full_text = f"Problem: {problem}\nSolution:\n{solution}"
        prompt_text = f"Problem: {problem}\nSolution:\n"

        # Tokenize prompt (to know its length)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # Tokenize full text
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Labels: copy input_ids, mask prompt tokens with -100
        labels = input_ids.copy()
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
def get_dataset():
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    # Filter GSM8K problems (7345 samples)
    dataset = dataset.filter(lambda x: x["source"] == "gsm8k")
    # Train/validation split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset["train"], dataset["test"]
    
def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

    # Pad all sequences to the longest in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./sft_qwen_gsm8k")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a DeepSpeed checkpoint directory to resume from")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialize DeepSpeed distributed backend
    deepspeed.init_distributed()

    # Prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # set pad token for causal LM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False   # mandatory when using gradient checkpointing

    # Load datasets
    train_data, val_data = get_dataset()
    train_dataset = SFTDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = SFTDataset(val_data, tokenizer, max_length=args.max_length)

    # Distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler,
        num_workers=4,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        pin_memory=True,
    )

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    # Scheduler (if not fully handled by DeepSpeed config)
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    if optimizer is not None and hasattr(optimizer, "optimizer"):
        # DeepSpeed wraps the optimizer; get internal one for scheduler
        base_optimizer = optimizer.optimizer
    else:
        base_optimizer = optimizer
    scheduler = get_linear_schedule_with_warmup(
        base_optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # TensorBoard (only on rank 0)
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    else:
        writer = None

    # Resume from checkpoint if requested
    resume_step = 0
    if args.resume_from_checkpoint:
        _, client_state = model_engine.load_checkpoint(args.resume_from_checkpoint)
        resume_step = client_state.get("global_step", 0)
        if dist.get_rank() == 0:
            print(f"Resumed from checkpoint at step {resume_step}")

    global_step = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()

        for step, batch in enumerate(train_loader):
            # Skip batches already covered by the resumed checkpoint
            if global_step < resume_step:
                global_step += 1
                continue

            # Move to device (DeepSpeed handles it automatically via model_engine)
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            outputs = model_engine(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["labels"])
            loss = outputs.loss
            loss_val = loss.item()
            model_engine.backward(loss)
            model_engine.step()
            del outputs, loss

            if scheduler:
                scheduler.step()

            global_step += 1

            # Save checkpoint (include global_step in client state for future resumes)
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                model_engine.save_checkpoint(args.output_dir, tag=f"step_{global_step}",
                                             client_state={"global_step": global_step})

            # Logging
            if dist.get_rank() == 0 and global_step % args.logging_steps == 0:
                writer.add_scalar("train/loss", loss_val, global_step)
                print(f"Step {global_step} | Train Loss: {loss_val:.4f}")

            # Periodically defragment CUDA memory (variable-length batches cause fragmentation)
            if global_step % args.logging_steps == 0:
                torch.cuda.empty_cache()

            # Validation
            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model_engine, val_loader, tokenizer)
                if dist.get_rank() == 0:
                    writer.add_scalar("val/loss", val_loss, global_step)
                    print(f"Step {global_step} | Validation Loss: {val_loss:.4f}")
                model_engine.train()  # switch back to train mode

    # Final save
    model_engine.save_checkpoint(args.output_dir, tag="final",
                                 client_state={"global_step": global_step})
    if dist.get_rank() == 0:
        writer.close()

def evaluate(model_engine, val_loader, tokenizer):
    model_engine.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            outputs = model_engine(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"],
                                   labels=batch["labels"])
            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
            count += batch["input_ids"].size(0)
    # Average loss across all processes
    total_loss_tensor = torch.tensor(total_loss).to(model_engine.device)
    count_tensor = torch.tensor(count).to(model_engine.device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    return (total_loss_tensor / count_tensor).item()

if __name__ == "__main__":
    main()
"""
Evaluate a model on GSM8K and MMLU benchmarks.

Aligns with lm_eval methodology where applicable.

Usage
-----
    # GSM8K (0-shot, official test set)
    python eval/evaluate.py --model_path ./sft_qwen_gsm8k/final --benchmark gsm8k

    # MMLU (5-shot, 57 subjects)
    python eval/evaluate.py --model_path ./sft_qwen_gsm8k/final --benchmark mmlu

    # Both benchmarks, limit samples for quick check
    python eval/evaluate.py --model_path ./sft_qwen_gsm8k/final --benchmark gsm8k mmlu \
        --gsm8k_max_samples 100 --mmlu_max_per_subject 20
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

CHOICE_LETTERS = ["A", "B", "C", "D"]

# All 57 MMLU subjects
MMLU_ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]


def _extract_boxed(text: str) -> str | None:
    """Extract the content of the last ``\\boxed{...}`` in *text*, or None."""
    # Match \boxed{...} allowing for nested braces (simple single-level)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


def _extract_last_number(text: str) -> str:
    """
    Fallback: find the last numeric pattern in *text* when no structured
    answer marker is present.

    Matches integers, decimals, fractions, and comma-separated numbers.
    Returns "" if nothing numeric is found.
    """
    # Match numbers like 42, 3.14, 1,200, 3/5, .5, -10
    pattern = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+|(?<!\d)\d+/\d+(?!\d)"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


def _normalize_number(s: str) -> str:
    """Normalize a numeric answer string for comparison."""
    s = s.strip().lower()
    s = s.replace("\\", "").replace("$", "").replace(",", "").replace("%", "").replace(" ", "")

    if "/" in s and s.count("/") == 1:
        try:
            num, denom = s.split("/")
            s = f"{float(num) / float(denom):.10g}"
        except (ValueError, ZeroDivisionError):
            pass

    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s = s[:-1]

    if s.startswith("."):
        s = "0" + s

    return s


def _is_deepspeed_checkpoint(path: str) -> bool:
    """Check whether *path* looks like a DeepSpeed (not HF) checkpoint."""
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        # DeepSpeed ZeRO optimizer state files: bf16_zero_pp_rank_*_optim_states.pt
        if "zero_pp_rank" in f and f.endswith("_optim_states.pt"):
            return True
    return False


def _load_model_from_deepspeed_ckpt(ckpt_path: str, base_model: str):
    """Load a HuggingFace model from a DeepSpeed ZeRO checkpoint."""
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    # DeepSpeed saves the 'latest' pointer in the parent output_dir, not in the
    # tag subdirectory.  Detect this: if ckpt_path/ has no 'latest' but its
    # parent does (pointing to ckpt_path's basename), pass (parent, tag).
    ckpt_dir = ckpt_path
    tag = None
    if not os.path.isfile(os.path.join(ckpt_path, "latest")):
        parent = os.path.dirname(ckpt_path)
        basename = os.path.basename(ckpt_path)
        latest_file = os.path.join(parent, "latest")
        if os.path.isfile(latest_file):
            with open(latest_file) as f:
                if f.read().strip() == basename:
                    ckpt_dir = parent
                    tag = basename

    print(f"Converting DeepSpeed checkpoint from {ckpt_path} ...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir, tag=tag)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("  Done.")
    return model


def load_model_and_tokenizer(model_path: str, base_model: str = "Qwen/Qwen2.5-1.5B"):
    """
    Load model weights and tokenizer.

    *model_path* may be an HF hub name, an HF-format checkpoint directory, or a
    DeepSpeed checkpoint directory.  The tokenizer is loaded from *model_path*
    if available, otherwise from *base_model*.
    """
    # ── Tokenizer ──────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (OSError, ValueError):
        print(f"Tokenizer not found in checkpoint, loading from {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────
    if _is_deepspeed_checkpoint(model_path):
        model = _load_model_from_deepspeed_ckpt(model_path, base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        model.eval()

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# GSM8K  (0-shot, official test set — 1319 examples)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gsm8k(model, tokenizer, max_samples=None, max_new_tokens=256, batch_size=16):
    """
    Evaluate GSM8K accuracy using the official test split.

    Processes prompts in batches for much faster generation throughput.
    Uses greedy decoding and extracts the answer after '####' from the
    generated solution. This matches the standard lm_eval GSM8K metric.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Pre-build all prompts and ground-truth answers
    prompts = []
    gt_answers = []
    for example in dataset:
        question = example["question"]
        gt_solution = example["answer"]
        prompts.append(f"Problem: {question}\nSolution:\n")
        gt_answers.append(
            gt_solution.split("####")[-1].strip() if "####" in gt_solution
            else gt_solution.strip()
        )

    correct, total = 0, 0
    no_marker_count = 0
    debug_samples = []

    # Left-padding is required for batched generation
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc="GSM8K"):
        batch_prompts = prompts[i:i + batch_size]
        batch_gt = gt_answers[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=768,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].size(1)   # padded length, same for all

        with torch.no_grad():
            gens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        for j, (gen, gt) in enumerate(zip(gens, batch_gt)):
            response_ids = gen[input_len:]                     # strip padding + prompt
            decoded = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Extract answer: try #### first, then \boxed{}, then numeric fallback
            if "####" in decoded:
                pred_answer = decoded.split("####")[-1].strip()
            elif (boxed := _extract_boxed(decoded)) is not None:
                pred_answer = boxed
            else:
                no_marker_count += 1
                pred_answer = _extract_last_number(decoded)

            if len(debug_samples) < 3:
                debug_samples.append((len(response_ids),
                                      decoded[-300:] if len(decoded) > 300 else decoded))

            if _normalize_number(pred_answer) == _normalize_number(gt):
                correct += 1
            total += 1

    tokenizer.padding_side = original_side

    acc = correct / total if total > 0 else 0.0
    print(f"GSM8K accuracy: {acc:.4f} ({correct}/{total})")
    if no_marker_count > 0:
        print(f"  ({no_marker_count}/{total} outputs missing '####' and '\\boxed{{}}' — used numeric fallback)")
    if debug_samples:
        print(f"  First {len(debug_samples)} outputs (gen_len, tail):")
        for gen_len, tail in debug_samples:
            print(f"    [{gen_len} tokens] ...{tail[-200:]}")

    return acc


# ═══════════════════════════════════════════════════════════════════════════
# MMLU  (5-shot, label-only logprob — matches lm_eval `acc` metric)
# ═══════════════════════════════════════════════════════════════════════════

def _format_mmlu_question(question: str, choices: list[str]) -> str:
    lines = [question]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def _mmulu_choice_token_ids(tokenizer) -> list[list[int]]:
    """
    Return candidate token IDs for each choice letter (A/B/C/D).

    Tries both ``"A"`` and ``" A"`` (space-prefixed) since BPE tokenizers
    differ in how they encode a letter immediately after a newline.
    Returns a list of lists: ``[[a_opts], [b_opts], [c_opts], [d_opts]]``.
    During scoring we take the max logprob across each letter's options.
    """
    ids = []
    for letter in CHOICE_LETTERS:
        options = []
        for prefix in ("", " "):
            tid = tokenizer.encode(prefix + letter, add_special_tokens=False)
            if len(tid) == 1:
                options.append(tid[0])
        if not options:
            raise RuntimeError(
                f"Cannot find single-token encoding for '{letter}'"
            )
        ids.append(options)
    return ids


def evaluate_mmlu(
    model,
    tokenizer,
    subjects=None,
    num_fewshot=5,
    max_samples_per_subject=None,
):
    """
    Evaluate MMLU multiple-choice accuracy.

    Uses *num_fewshot* examples from the 'dev' split (5 per subject), formatted
    with the correct answer letter.  Scores each test question by computing
    log P(answer_letter | prompt) and picking the highest.

    This is equivalent to lm_eval's ``mmlu`` task with ``acc`` metric.
    """
    if subjects is None:
        subjects = MMLU_ALL_SUBJECTS

    dev_set = load_dataset("cais/mmlu", "all", split="dev")
    test_set = load_dataset("cais/mmlu", "all", split="test")

    # Pre-compute choice token IDs (depends only on the tokenizer)
    choice_token_ids = _mmulu_choice_token_ids(tokenizer)

    all_correct, all_total = 0, 0
    subject_accuracies = {}

    for subject in subjects:
        # Few-shot examples for this subject
        dev_subject = dev_set.filter(lambda x: x["subject"] == subject)
        dev_examples = list(dev_subject.select(range(min(num_fewshot, len(dev_subject)))))

        # Test examples for this subject
        test_subject = test_set.filter(lambda x: x["subject"] == subject)
        if max_samples_per_subject is not None:
            test_subject = test_subject.select(
                range(min(max_samples_per_subject, len(test_subject)))
            )

        correct, total = 0, 0
        for example in tqdm(test_subject, desc=f"MMLU/{subject}"):
            question = example["question"]
            choices = example["choices"]
            answer_idx = example["answer"]

            # Build few-shot prompt
            parts = []
            for dev_ex in dev_examples:
                parts.append(_format_mmlu_question(dev_ex["question"], dev_ex["choices"]))
                parts.append(f"Answer: {CHOICE_LETTERS[dev_ex['answer']]}\n")

            test_question_block = _format_mmlu_question(question, choices)
            parts.append(test_question_block)
            parts.append("Answer:")

            prompt = "\n".join(parts)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                next_logits = outputs.logits[0, -1, :].float() # one pass only (prefill)
                next_logprobs = F.log_softmax(next_logits, dim=-1)

            # Score each choice by max logprob across its candidate token IDs
            choice_logprobs = [
                max(next_logprobs[tid].item() for tid in token_options)
                for token_options in choice_token_ids
            ]
            pred_idx = int(torch.argmax(torch.tensor(choice_logprobs)).item())

            if pred_idx == answer_idx:
                correct += 1
            total += 1

        subj_acc = correct / total if total > 0 else 0.0
        subject_accuracies[subject] = subj_acc
        all_correct += correct
        all_total += total
        print(f"  {subject}: {subj_acc:.4f} ({correct}/{total})")

    overall_acc = all_correct / all_total if all_total > 0 else 0.0
    print(f"\nMMLU overall accuracy: {overall_acc:.4f} ({all_correct}/{all_total})")
    return overall_acc, subject_accuracies


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a model on GSM8K and MMLU")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B",
                   help="Model checkpoint path, DeepSpeed ckpt dir, or HF hub name")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B",
                   help="Base model for architecture and tokenizer (used when --model_path is a DeepSpeed ckpt)")
    p.add_argument("--benchmark", type=str, nargs="+",
                   default=["gsm8k"],
                   choices=["gsm8k", "mmlu"],
                   help="Benchmarks to evaluate (default: gsm8k)")
    p.add_argument("--gsm8k_max_samples", type=int, default=None,
                   help="Limit GSM8K samples (default: all 1319)")
    p.add_argument("--gsm8k_batch_size", type=int, default=32,
                   help="Batch size for GSM8K generation (default: 32)")
    p.add_argument("--mmlu_max_per_subject", type=int, default=None,
                   help="Limit MMLU samples per subject (default: all)")
    p.add_argument("--mmlu_subjects", type=str, nargs="*", default=None,
                   help="MMLU subjects to evaluate (default: all 57)")
    p.add_argument("--mmlu_fewshot", type=int, default=5,
                   help="Number of few-shot examples for MMLU (default: 5)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens for GSM8K generation")
    p.add_argument("--output_file", type=str, default="results.json",
                   help="Save results as JSON to this path (default: results.json)")
    return p.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)

    results = {}

    if "gsm8k" in args.benchmark:
        print("\n" + "=" * 60)
        print("GSM8K  (0-shot, official test set)")
        print("=" * 60)
        results["gsm8k_accuracy"] = evaluate_gsm8k(
            model, tokenizer,
            max_samples=args.gsm8k_max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.gsm8k_batch_size,
        )

    if "mmlu" in args.benchmark:
        print("\n" + "=" * 60)
        print(f"MMLU  ({args.mmlu_fewshot}-shot, {len(args.mmlu_subjects or MMLU_ALL_SUBJECTS)} subjects)")
        print("=" * 60)
        overall, per_subject = evaluate_mmlu(
            model, tokenizer,
            subjects=args.mmlu_subjects,
            num_fewshot=args.mmlu_fewshot,
            max_samples_per_subject=args.mmlu_max_per_subject,
        )
        results["mmlu_accuracy"] = overall
        results["mmlu_per_subject"] = per_subject

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in results.items():
        if k == "mmlu_per_subject":
            continue
        print(f"  {k}: {v:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    if args.output_file:
        output = {
            "model_path": args.model_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmarks": args.benchmark,
            "results": {},
        }
        if "gsm8k_accuracy" in results:
            output["results"]["gsm8k"] = {
                "accuracy": results["gsm8k_accuracy"],
                "num_fewshot": 0,
                "dataset": "openai/gsm8k (test)",
            }
        if "mmlu_accuracy" in results:
            output["results"]["mmlu"] = {
                "accuracy": results["mmlu_accuracy"],
                "num_fewshot": args.mmlu_fewshot,
                "num_subjects": len(args.mmlu_subjects or MMLU_ALL_SUBJECTS),
                "per_subject": results.get("mmlu_per_subject", {}),
            }

        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()

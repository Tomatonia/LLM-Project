"""
Reward functions for GSM8K mathematical reasoning.

GRPO uses per-response scalar rewards. For GSM8K, the reward is based on
whether the model's final answer matches the ground-truth answer.
"""

import re


def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the final answer from a GSM8K-style solution.

    The GSM8K format expects the final answer after '####' (4 hash marks).
    Returns the stripped text after the last '####', or None if not found.
    """
    if "####" in text:
        return text.split("####")[-1].strip()
    return None


def normalize_number(s: str) -> str:
    """
    Normalize a numeric string for comparison.

    Handles: '1,200', '$5.00', '60%', '5.0', '5.250', '1/2', '.5', '5.'
    """
    s = s.strip().lower()
    s = s.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")

    # Fraction → decimal  (e.g. "1/2" → "0.5", "3/5" → "0.6")
    if "/" in s and s.count("/") == 1:
        try:
            num, denom = s.split("/")
            val = float(num) / float(denom)
            s = f"{val:.10g}"
        except (ValueError, ZeroDivisionError):
            pass

    # Strip trailing zeros after decimal  ("5.250" → "5.25", "5.00" → "5")
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s = s[:-1]

    # Normalise missing leading zero  (".5" → "0.5")
    if s.startswith("."):
        s = "0" + s

    return s


def gsm8k_reward(generated_text: str, ground_truth_answer: str) -> float:
    """
    Compute a binary reward: 1.0 if the extracted answer matches the ground truth.

    Args:
        generated_text: The full model-generated solution (prompt + completion).
        ground_truth_answer: The expected answer string (from the dataset).

    Returns:
        1.0 if answers match, 0.0 otherwise.
    """
    pred = extract_gsm8k_answer(generated_text)
    if pred is None:
        return 0.0
    return 1.0 if normalize_number(pred) == normalize_number(ground_truth_answer) else 0.0


def extract_gt_answer_from_solution(solution_text: str) -> str:
    """
    Extract the ground-truth answer from a GSM8K solution string.

    The solution typically ends with '#### <number>'.
    """
    return extract_gsm8k_answer(solution_text) or solution_text.strip()

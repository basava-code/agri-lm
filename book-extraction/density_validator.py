"""Quality Gate for Text Density Validation.

Filters out low-quality text passages before they enter the CPT output.
Thresholds are representation-aware: fact-dense representations
(ledgers, comparison matrices, exam banks, chemical chains) legitimately
carry high digit ratios and short formula tokens, so they get relaxed caps.
"""

from typing import TypedDict

FACT_DENSE_REPRESENTATIONS = {
    "quant_fact_ledger",
    "numeric_comparison_matrix",
    "causal_chemical_chain",
    "exam_bank_grounded",
    "technical_factsheet",
}


class ValidationResult(TypedDict):
    passed: bool
    reason: str
    alpha_ratio: float
    digit_ratio: float
    avg_word_len: float

def validate_text(text: str, representation_type: str | None = None) -> ValidationResult:
    """
    Checks a text passage for signs of extraction garbage:
    - Too many digits (e.g. table data extracted as strings)
    - Too many special characters (PDF noise)
    - Avg word length too short (broken words / OCR errors)
    - Alpha ratio below threshold
    """
    fact_dense = representation_type in FACT_DENSE_REPRESENTATIONS
    digit_cap = 0.45 if fact_dense else 0.20
    min_word_len = 2.5 if fact_dense else 3.0
    alpha_floor = 0.45 if fact_dense else 0.70

    words = text.split()
    if not words:
        return {
            "passed": False,
            "reason": "empty_text",
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "avg_word_len": 0.0
        }

    alpha_chars = sum(c.isalpha() for c in text)
    digit_chars = sum(c.isdigit() for c in text)
    total_chars = max(len(text), 1)
    avg_word_len = sum(len(w) for w in words) / len(words)

    alpha_ratio = alpha_chars / total_chars
    digit_ratio = digit_chars / total_chars

    if alpha_ratio < alpha_floor:
        return {
            "passed": False,
            "reason": f"alpha_ratio={alpha_ratio:.2f} < {alpha_floor:.2f}",
            "alpha_ratio": alpha_ratio,
            "digit_ratio": digit_ratio,
            "avg_word_len": avg_word_len
        }

    if digit_ratio > digit_cap:
        return {
            "passed": False,
            "reason": f"digit_ratio={digit_ratio:.2f} > {digit_cap:.2f} (rep={representation_type})",
            "alpha_ratio": alpha_ratio,
            "digit_ratio": digit_ratio,
            "avg_word_len": avg_word_len
        }

    if not (min_word_len <= avg_word_len <= 8.0):
        return {
            "passed": False,
            "reason": f"avg_word_len={avg_word_len:.1f} out of [{min_word_len},8]",
            "alpha_ratio": alpha_ratio,
            "digit_ratio": digit_ratio,
            "avg_word_len": avg_word_len
        }

    return {
        "passed": True,
        "reason": "",
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "avg_word_len": avg_word_len
    }

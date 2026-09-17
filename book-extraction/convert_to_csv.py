#!/usr/bin/env python3
"""Convert a Gemini QnA JSONL dataset to a simple CSV.

Each line of the input JSONL file looks like:
```
{"messages": [{"role": "user", "content": "question..."},
                {"role": "assistant", "content": "answer..."}]}
```
The script extracts the ``user`` content as the *question* and the ``assistant``
content as the *answer* and writes a CSV with two columns: ``question`` and ``answer``.

Usage:
    python convert_to_csv.py [--input INPUT_JSONL] [--output OUTPUT_CSV]

If no arguments are supplied the script defaults to the project's processed
data paths:
    INPUT  = data/processed/gemma_qna_dataset.jsonl
    OUTPUT = data/processed/gemma_qna_dataset.csv

The script is deliberately lightweight and does not depend on any third‑party
libraries – only the Python standard library – so it can be run in any
environment (including the project's existing virtualenv).
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gemini QnA JSONL to CSV (question, answer).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parents[1] / "data" / "processed" / "gemma_qna_dataset.jsonl"),
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parents[1] / "data" / "processed" / "pp_khrif_qna_dataset.csv"),
        help="Destination CSV file.",
    )
    return parser.parse_args()


def extract_qa(line: str) -> tuple[str, str] | None:
    """Parse a single JSONL line and return a (question, answer) pair.

    The function is tolerant of missing roles – if either the ``user`` or
    ``assistant`` role is not present the line is ignored.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"Skipping malformed JSON line: {exc}", file=sys.stderr)
        return None

    messages = data.get("messages", [])
    question = answer = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user" and question is None:
            question = content.strip()
        elif role == "assistant" and answer is None:
            answer = content.strip()
        if question and answer:
            break

    if question is None or answer is None:
        print("Incomplete QnA pair encountered; skipping.", file=sys.stderr)
        return None
    return question, answer


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as fout:
        writer = csv.writer(fout)
        writer.writerow(["question", "answer"])  # header
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            qa = extract_qa(line)
            if qa:
                writer.writerow(qa)
            else:
                print(f"Line {lineno} could not be converted.", file=sys.stderr)

    print(f"Conversion completed. CSV saved to: {output_path}")


if __name__ == "__main__":
    main()

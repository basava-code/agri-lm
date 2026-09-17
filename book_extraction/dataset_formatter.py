"""Dataset Formatter for Agri-LM QnA Variations.

This module processes the raw JSON Lines results downloaded from the Gemini Batch API,
unpacks the candidate response JSON, flattens the nested crop QnA variations,
and formats them into clean conversational user-assistant message turns (Option B)
ready for small model (Gemma) fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import re



def find_project_root() -> Path:
    """Dynamically resolve the project root by searching upwards."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".env").exists() or (parent / "CLAUDE.md").exists():
            return parent
    return current


PROJECT_ROOT = find_project_root()
DEFAULT_METADATA_JSON = PROJECT_ROOT / "data" / "processed" / "default_batch_metadata.json"
DEFAULT_RESULTS_JSONL = PROJECT_ROOT / "data" / "processed" / "default_batch_results.jsonl"
DEFAULT_DATASET_JSONL = PROJECT_ROOT / "data" / "processed" / "default_gemma_qna_dataset.jsonl"


def format_dataset(
    results_path: Path = DEFAULT_RESULTS_JSONL,
    output_path: Path = DEFAULT_DATASET_JSONL,
    schema_version: str = "v1"
) -> tuple[int, int]:
    """Parses raw batch results, flattens QnA variations, and saves them in chat format.

    Returns a tuple of (total_extracted_facts, total_flattened_samples).
    """
    if not results_path.exists():
        raise FileNotFoundError(
            f"Raw batch results file not found at: {results_path}.\n"
            "Please run batch_generator.py to generate results first."
        )

    total_facts = 0
    total_samples = 0
    chapter_stats: dict[str, int] = {}

    print(f"Reading batch results from: {results_path}")
    print(f"Writing final fine-tuning dataset to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                continue

            key = data.get("key", f"line_{line_num}")
            
            # Check for API-level request errors
            if "error" in data:
                print(f"Request with key '{key}' failed with API error: {data['error']}", file=sys.stderr)
                continue

            response = data.get("response", {})
            candidates = response.get("candidates", [])
            if not candidates:
                print(f"Warning: No candidates found for key '{key}'", file=sys.stderr)
                continue

            # Extract the raw text from candidate parts
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                print(f"Warning: No content parts found for key '{key}'", file=sys.stderr)
                continue

            raw_text = parts[0].get("text", "")
            if not raw_text:
                print(f"Warning: Empty text field for key '{key}'", file=sys.stderr)
                continue

            # Clean markdown code blocks (e.g. ```json ... ```) if emitted by the model
            cleaned_json_str = raw_text.strip()
            if cleaned_json_str.startswith("```"):
                # Remove starting markdown block
                cleaned_json_str = re.sub(r"^```(?:json)?\n", "", cleaned_json_str)
                # Remove ending markdown block
                cleaned_json_str = re.sub(r"\n```$", "", cleaned_json_str)

            try:
                # Parse the Structured Output JSON matching CropQnADataset schema
                extracted_data = json.loads(cleaned_json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse structured JSON from key '{key}': {e}", file=sys.stderr)
                # Log snippet for debugging
                print(f"Raw text snippet: {raw_text[:200]}...", file=sys.stderr)
                continue

            qna_groups = extracted_data.get("qna_groups", [])
            chapter_samples_count = 0

            for group in qna_groups:
                total_facts += 1
                
                # Auto-detect schema: new schema has 'authoritative_answer' (or 'answer' for the simplified 4B prompt)
                if "authoritative_answer" in group or "answer" in group:
                    # Single authoritative answer
                    answer = group.get("authoritative_answer", group.get("answer", "")).strip()
                    
                    # Alternative questions (array) or a single question
                    if "alternative_questions" in group:
                        questions = group.get("alternative_questions", [])
                    else:
                        single_q = group.get("question", "")
                        questions = [single_q] if single_q else []
                    
                    if not answer or not questions:
                        continue
                        
                    for question in questions:
                        question = question.strip()
                        if not question:
                            continue
                            
                        chat_sample = {
                            "messages": [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                        }
                        f_out.write(json.dumps(chat_sample, ensure_ascii=False) + "\n")
                        total_samples += 1
                        chapter_samples_count += 1
                else:
                    # Original schema: Nested question-answer variations list
                    variations = group.get("variations", [])
                    
                    for var in variations:
                        question = var.get("question", "").strip()
                        answer = var.get("answer", "").strip()

                        if not question or not answer:
                            continue


                        chat_sample = {
                            "messages": [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                        }

                        # Save as a single line in JSONL file
                        f_out.write(json.dumps(chat_sample, ensure_ascii=False) + "\n")
                        total_samples += 1
                        chapter_samples_count += 1

            print(f"[{key:<22}] Extracted {len(qna_groups):>3} facts → {chapter_samples_count:>4} training pairs")
            chapter_stats[key] = chapter_samples_count

    print("\n" + "="*50)
    print("FINE-TUNING DATASET EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total logical agricultural facts extracted: {total_facts}")
    print(f"Total flattened QnA training samples generated: {total_samples}")
    print(f"Target fine-tuning file: {output_path}")
    print("\nCrop-wise sample distribution:")
    for crop, count in sorted(chapter_stats.items()):
        print(f" - {crop:<25}: {count:>5} samples")
    print("="*50)

    return total_facts, total_samples


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Format Raw Batch Results into Gemma Fine-tuning Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--format",
        choices=["v1", "v2"],
        default="v1",
        help="Specify the pipeline's output schema version: 'v1' for nested variations (original), 'v2' for single answer with alternative questions (refactored)."
    )
    args = parser.parse_args()

    try:
        format_dataset(DEFAULT_RESULTS_JSONL, DEFAULT_DATASET_JSONL, schema_version=args.format)
    except Exception as e:
        print(f"Error during dataset formatting: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

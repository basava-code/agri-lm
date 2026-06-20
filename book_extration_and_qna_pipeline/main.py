#!/usr/bin/env python3
"""Centralized Pipeline Orchestrator for Agri-LM Crop practices PDF extraction.

Runs all 5 stages in sequence:
1. TOC Detection (toc_detector)
2. TOC Extraction & Page Alignment (toc_extractor)
3. PDF Chapter Chunking (chunker)
4. Async local QnA generation (batch_generator)
5. Fine-tuning dataset formatting (dataset_formatter)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name in ("src", "book_data_extractor") else current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from book_data_extractor.toc_detector import detect_toc
from book_data_extractor.toc_extractor import extract_and_solve_toc
from book_data_extractor.chunker import extract_chapters
from book_data_extractor.batch_generator import run_batch_generator
from book_data_extractor.dataset_formatter import format_dataset


async def async_pipeline(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path

    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    book_name = pdf_path.stem
    print(f"============================================================")
    print(f"🏁 Starting Book Data Extraction Pipeline for: {pdf_path.name}")
    print(f"============================================================")

    start_time = time.time()

    # Stage 1: TOC Detection
    print("\n--- [Stage 1/5] Auto-detecting Table of Contents boundaries ---")
    if args.dry_run:
        print("[Dry Run] Skipping TOC Detection scan.")
        detected_toc_pages = [3, 4]
        detected_highest_printed_page = args.actual_pages or 200
        detected_title = "Dry Run Book Title"
    else:
        try:
            toc_res = detect_toc(
                pdf_path=pdf_path,
                pages_to_scan=args.pages_to_scan,
                model_name=args.model
            )
            detected_toc_pages = toc_res.toc_pages
            detected_highest_printed_page = toc_res.highest_printed_page
            detected_title = toc_res.title
            print(f" -> Detected Book Title: '{detected_title}'")
            print(f" -> Detected TOC pages: {detected_toc_pages}")
            print(f" -> Detected highest printed page: {detected_highest_printed_page}")
        except Exception as e:
            print(f"Error in TOC Detection: {e}", file=sys.stderr)
            sys.exit(1)

    # Use args.actual_pages if supplied, otherwise fallback to detected highest printed page
    actual_pages = args.actual_pages or detected_highest_printed_page
    if not actual_pages:
        print("Error: Could not resolve total printed pages. Please supply --actual-pages.", file=sys.stderr)
        sys.exit(1)

    # Stage 2: TOC Extraction & Page Alignment
    print("\n--- [Stage 2/5] Running TOC Extraction & Page Alignment ---")
    toc_json_path = project_root / "data" / "processed" / f"{book_name}_toc.json"
    if args.dry_run:
        print(f"[Dry Run] Skipping LLM extraction. Solved TOC JSON would be written to: {toc_json_path}")
    else:
        try:
            resolved_chapters = extract_and_solve_toc(
                pdf_path=pdf_path,
                actual_pages=actual_pages,
                toc_pages=detected_toc_pages,
                model_name=args.model
            )
            # Save resolved boundaries to json
            toc_json_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(toc_json_path, "w", encoding="utf-8") as f:
                json.dump(resolved_chapters, f, indent=4, ensure_ascii=False)
            print(f" -> Solved TOC saved to: {toc_json_path}")
        except Exception as e:
            print(f"Error in TOC Extraction & Solving: {e}", file=sys.stderr)
            sys.exit(1)

    # Stage 3: Chapter Chunker
    print("\n--- [Stage 3/5] Chunking PDF Chapters ---")
    output_chapters_json = project_root / "data" / "processed" / f"{book_name}_chapters.json"
    if args.dry_run:
        print(f"[Dry Run] Skipping PDF extraction. Chapters would be saved to: {output_chapters_json}")
    else:
        try:
            extract_chapters(
                pdf_path=pdf_path,
                toc_json_path=toc_json_path,
                output_chapters_json=output_chapters_json
            )
            print(f" -> Chapters extracted successfully.")
        except Exception as e:
            print(f"Error in Chapter Chunking: {e}", file=sys.stderr)
            sys.exit(1)

    # Stage 4: Async QnA Generation
    print("\n--- [Stage 4/5] Spawning Async QnA Generation ---")
    if args.dry_run:
        print("[Dry Run] Skipping local LLM async spawning.")
    else:
        try:
            await run_batch_generator(
                input_chunks_path=output_chapters_json,
                output_results_path=project_root / "data" / "processed" / f"{book_name}_batch_results.jsonl",
                model_name=args.model,
                prompt_version=args.prompt_version,
                max_concurrency=args.max_concurrency,
                dry_run=False
            )
            print(" -> Async batch generation completed.")
        except Exception as e:
            print(f"Error in QnA Batch Generation: {e}", file=sys.stderr)
            sys.exit(1)

    # Stage 5: Dataset Formatting
    print("\n--- [Stage 5/5] Formatting QnA Dataset ---")
    results_path = project_root / "data" / "processed" / f"{book_name}_batch_results.jsonl"
    dataset_output_path = project_root / "data" / "processed" / f"{book_name}_gemma_qna_dataset.jsonl"
    legacy_dataset_path = project_root / "data" / "processed" / f"{book_name}_legacy_gemma_qna_dataset.jsonl"
    
    if args.dry_run:
        print(f"[Dry Run] Skipping formatting. Formatted dataset would be saved to: {dataset_output_path}")
    else:
        try:
            facts, samples = format_dataset(
                results_path=results_path,
                output_path=dataset_output_path,
                schema_version=args.format
            )
            # Duplicate output to legacy path to match Downstream expectations
            import shutil
            shutil.copyfile(dataset_output_path, legacy_dataset_path)
            print(f" -> Success! Extracted {facts} core facts into {samples} fine-tuning conversational turns.")
            print(f" -> Saved dataset to: {dataset_output_path}")
            print(f" -> Legacy dataset duplicate: {legacy_dataset_path}")
        except Exception as e:
            print(f"Error during dataset formatting: {e}", file=sys.stderr)
            sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n============================================================")
    print(f"🎉 Pipeline finished successfully in {elapsed:.1f} seconds.")
    print(f"============================================================")


def main() -> None:
    # Find project root for loading env
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent if current_dir.name in ("src", "book_data_extractor") else current_dir
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(
        description="Unified End-to-End PDF to QnA Dataset Extraction Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=str,
        help="Path to the source PDF book file to process."
    )
    parser.add_argument(
        "--actual-pages",
        type=int,
        default=None,
        help="Total printed/last-numbered page count in the book. If not supplied, defaults to the page count auto-detected by Stage 1."
    )
    parser.add_argument(
        "--pages-to-scan",
        type=int,
        default=15,
        help="Number of pages to scan at the start of the book for TOC detection."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override to use for TOC detection, TOC extraction, and batch QnA generation."
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v3",
        help="Prompt configuration version (v1, v2, v3) from prompts.py to guide the extraction."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent async LLM queries during generation."
    )
    parser.add_argument(
        "--format",
        choices=["v1", "v2"],
        default="v1",
        help="The final dataset conversation formatting scheme version ('v1' or 'v2')."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations and dry run the execution steps without calling LLM endpoints or modifying documents."
    )

    args = parser.parse_args()

    try:
        asyncio.run(async_pipeline(args))
    except KeyboardInterrupt:
        print("\nPipeline execution cancelled by operator.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

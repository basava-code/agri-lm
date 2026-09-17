"""Async Local Batch Generator for Agri-LM QnA Extraction.

This module processes chunked chapters in parallel using LangChain's async API,
respects rate-limits via concurrency throttling, supports self-healing resumption,
and iteratively saves completed responses to a local JSONL file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name in ("src", "book_data_extractor") else current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from book_data_extractor.llm_factory import get_llm
from book_data_extractor.prompts import CropQnADataset, get_prompt


def find_project_root() -> Path:
    """Dynamically resolve the project root by searching upwards."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".env").exists() or (parent / "CLAUDE.md").exists():
            return parent
    return current


PROJECT_ROOT = find_project_root()
DEFAULT_INPUT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "target_book_chapters.json"
DEFAULT_RESULTS_JSONL = PROJECT_ROOT / "data" / "processed" / "target_book_batch_results.jsonl"


async def process_chunk(
    ch: dict[str, str],
    sem: asyncio.Semaphore,
    chain,
    user_template: str,
    results_file: Path
) -> None:
    """Processes a single chapter chunk with concurrency throttling and retries."""
    async with sem:
        key = ch["key"]
        title = ch["title"]
        text = ch["text"]
        user_content = user_template.format(title=title, key=key, text=text)

        print(f"[Async Generator] Processing chunk '{key}' ({title})...")

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True
            ):
                with attempt:
                    result = await asyncio.to_thread(chain.invoke, {"text_content": user_content})
                    # Salvage root keys if the parser generated a list under the wrong name
                    if "qna_groups" not in result:
                        for v in result.values():
                            if isinstance(v, list):
                                result["qna_groups"] = v
                                break
                    validated_result = CropQnADataset(**result)

            print(f"[Async Generator] Successfully processed chunk '{key}'. Saving result...")
            
            # Convert response structure to match Gemini Batch API Candidates format for dataset_formatter compatibility
            json_text = json.dumps(validated_result.model_dump(), ensure_ascii=False)
            row_data = {
                "key": key,
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json_text
                                    }
                                ]
                            }
                        }
                    ]
                }
            }

            # Append the completed row to the results file
            with open(results_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Async Generator] Error processing chunk '{key}': {e}", file=sys.stderr)
            raise e


async def run_batch_generator(
    input_chunks_path: Path,
    output_results_path: Path,
    model_name: str | None = None,
    prompt_version: str = "v3",
    max_concurrency: int = 1,
    dry_run: bool = False
) -> None:
    """Processes chunked chapters in parallel using LangChain's async API,
    respecting rate-limits via concurrency throttling, and saves results locally.
    """
    # Load prompt template and system instructions
    prompt_cfg = get_prompt(prompt_version)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_core.output_parsers import JsonOutputParser

    if isinstance(prompt_cfg, ChatPromptTemplate):
        system_instruction = prompt_cfg.messages[0].prompt.template
        user_template = prompt_cfg.messages[1].prompt.template
    else:
        system_instruction = prompt_cfg["system_instruction"]
        user_template = prompt_cfg["user_template"]

    # Read the input chapters chunks
    if not input_chunks_path.exists():
        raise FileNotFoundError(
            f"Chunked chapters file not found at: {input_chunks_path}. "
            "Please run chunker.py first to generate it."
        )

    with open(input_chunks_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)

    # Self-healing resumption: identify already completed keys
    completed_keys = set()
    if output_results_path.exists():
        with open(output_results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_keys.add(data["key"])
                    except Exception:
                        pass

    pending_chapters = [ch for ch in chapters if ch["key"] not in completed_keys]
    print(f"[Async Generator] Total chapters: {len(chapters)} | Completed: {len(completed_keys)} | Pending: {len(pending_chapters)}")

    if dry_run:
        print("\n=== DRY RUN COMPLETED ===")
        print(f"Verified {len(pending_chapters)} pending chunks to process.")
        print(f"Dry run finished successfully.")
        return

    if not pending_chapters:
        print("[Async Generator] 🎉 All chapters have already been processed successfully!")
        return

    # Initialize LangChain LLM
    print(f"\n[Async Generator] Initialising LLM (model override: {model_name or 'default'})...")
    llm = get_llm(model_name=model_name, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=CropQnADataset)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction + "\n\n{format_instructions}"),
        ("human", "{text_content}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    # Setup semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"[Async Generator] Starting async processing with max_concurrency={max_concurrency}...")

    start_time = time.time()
    
    # Spawn and run tasks
    tasks = []
    for ch in pending_chapters:
        tasks.append(process_chunk(ch, semaphore, chain, user_template, output_results_path))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"A chunk failed completely: {result}")

    elapsed_time = time.time() - start_time
    print(f"\n🎉 [Async Generator] Processing completed in {elapsed_time:.1f} seconds.")
    print(f"Results saved to: {output_results_path}")


async def async_main() -> None:
    # Load environment variables from .env
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Agri-LM QnA Extraction Async Local Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_CHUNKS,
        help="Path to the input JSON file containing chunks."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_RESULTS_JSONL,
        help="Path to the output JSONL file to store results."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override to use. If None, uses LLM_PROVIDER's default model."
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v3",
        help="Prompt version from prompts.py (e.g. v1, v2, v3)."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent async API requests."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: verify configuration and list pending chunks without making requests."
    )

    args = parser.parse_args()
    await run_batch_generator(
        input_chunks_path=args.input_file,
        output_results_path=args.output_file,
        model_name=args.model,
        prompt_version=args.prompt_version,
        max_concurrency=args.max_concurrency,
        dry_run=args.dry_run
    )


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nExecution interrupted by operator. Completed progress remains saved in results JSONL.")
        sys.exit(1)


if __name__ == "__main__":
    main()

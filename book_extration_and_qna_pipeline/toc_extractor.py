"""Dynamic TOC Extractor and Page Alignment Solver.

This module provides a generic, book-agnostic pipeline that reads any PDF,
calculates the page offset between printed numbers and PDF page indices,
extracts raw text from specified TOC pages, and calls LangChain LLM with Pydantic
Structured Outputs to parse the table of contents.

It then programmatically solves the adjacent page boundaries of all chapters,
applies the calculated offset to align them with actual 1-indexed PDF page ranges,
and saves the resolved configuration under the book's real name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from pypdf import PdfReader
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "book_data_extractor" else current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from book_data_extractor.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser


class SubTOCEntry(BaseModel):
    """Structured database record for a raw Table of Contents entry."""
    title: str = Field(
        description="The full, official title of this subtopic exactly as listed in the Table of Contents."
    )
    start_page: int = Field(
        description="The starting printed page number of this subtopic (as listed in the Table of Contents)."
    )

class TOCEntry(BaseModel):
    """Structured database record for a raw Table of Contents entry."""
    unique_key: str = Field(
        description="A clean, unique snake_case string identifier for this chapter (e.g. '01_cereals_rice', '04_cereals_maize')."
    )
    title: str = Field(
        description="The full, official title of this chapter exactly as listed in the Table of Contents."
    )
    start_page: int = Field(
        description="The starting printed page number of this chapter (as listed in the Table of Contents)."
    )
    topics: list[SubTOCEntry] = Field(
        description="A complete, sequential list of all subtopics and sections extracted from this chapter's Table of Contents."
    )


class RawTOCDataset(BaseModel):
    """The master Pydantic container schema for all extracted TOC entries."""
    entries: list[TOCEntry] = Field(
        description="A complete, sequential list of all chapters and sections extracted from the Table of Contents."
    )


def extract_raw_toc_text(pdf_path: Path, toc_pages: list[int]) -> str:
    """Extracts raw text from the specified 1-indexed PDF page list."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    extracted_texts = []
    print(f"Extracting raw TOC text from PDF pages: {toc_pages}")
    
    for page_num in toc_pages:
        idx = page_num - 1
        if idx < 0 or idx >= total_pages:
            print(f"Warning: Specified TOC page {page_num} is out of bounds (1 to {total_pages}). Skipping.", file=sys.stderr)
            continue
            
        page = reader.pages[idx]
        raw_text = page.extract_text() or ""
        if raw_text.strip():
            extracted_texts.append(f"--- PDF Page {page_num} ---\n{raw_text.strip()}")
            
    return "\n\n".join(extracted_texts)


def solve_chapter_boundaries(
    raw_dataset: RawTOCDataset,
    actual_pages: int,
    offset: int
) -> dict[str, dict[str, any]]:
    """Sorts starting pages, programmatically solves adjacent chapter bounds,
    and shifts all indices by the dynamic PDF offset to return final ranges.
    """
    entries = sorted(raw_dataset.entries, key=lambda x: x.start_page)
    
    resolved_chapters = {}
    
    for i, entry in enumerate(entries):
        key = entry.unique_key
        title = entry.title
        start_page = entry.start_page
        
        if i < len(entries) - 1:
            end_page = entries[i+1].start_page - 1
        else:
            end_page = actual_pages
            
        # Validate that the boundary is logical
        if end_page < start_page:
            end_page = start_page
            
        pdf_start = start_page + offset
        pdf_end = end_page + offset
        topics = {}
        for idx_t, topic in enumerate(entry.topics):
            start_page_topic = topic.start_page + offset
            if idx_t < len(entry.topics) - 1:
                end_page_topic = entry.topics[idx_t+1].start_page + offset
            else:
                end_page_topic = start_page_topic
            topics[topic.title] = {
                "range": [start_page_topic, end_page_topic],
                "title": topic.title
            }
        
        resolved_chapters[key] = {
            "range": [pdf_start, pdf_end],
            "title": title,
            "topics": topics
        }
        
    return resolved_chapters


def extract_and_solve_toc(
    pdf_path: Path,
    actual_pages: int,
    toc_pages: list[int],
    model_name: str | None = None
) -> dict[str, dict[str, any]]:
    """Extracts raw TOC text, queries the LLM with structured outputs,
    and programmatically solves the aligned chapter boundaries.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    total_pdf_pages = len(reader.pages)
    offset = total_pdf_pages - actual_pages
    
    print(f"Processing target book: {pdf_path.name}")
    print(f"Total PDF pages: {total_pdf_pages}")
    print(f"Total printed page count: {actual_pages}")
    print(f"Resolved dynamic PDF offset: {offset:+} pages")

    # Extract raw TOC text from PDF pages
    raw_toc_text = extract_raw_toc_text(pdf_path, toc_pages)

    # Setup the prompt to extract chapters
    system_instruction = (
        "You are an expert Document Layout Analyst and Data Structuring Engine. "
        "Your primary function is to parse raw, unstructured Table of Contents (TOC) text extracted from PDFs "
        "(often containing OCR noise, dotted leaders, or irregular spacing) and convert it into a strictly hierarchical, "
        "machine-readable JSON dataset.\n\n"

        "### CORE OBJECTIVE\n"
        "Extract a two-level hierarchy: Main Chapters/Sections (`TOCEntry`) and their nested Subtopics (`SubTOCEntry`). "
        "Preserve the exact semantic hierarchy and ensure all page numbers are valid integers.\n\n"

        "### PROCESSING GUIDELINES\n\n"

        "1. HIERARCHICAL EXTRACTION:\n"
        "   - Identify main chapters/sections that act as parent containers. Assign these to `TOCEntry`.\n"
        "   - Identify all nested sub-sections, topics, or thematic divisions belonging to each chapter. Assign these to the `topics` list within the corresponding `TOCEntry`.\n"
        "   - If a section has no subtopics, set `topics` to an empty list `[]`. Do NOT omit it.\n"
        "   - Ignore standalone headers or entries without page numbers.\n"
        "   - Use indentation, numbering patterns (e.g., 1.1, 1.2), or visual cues in the raw text to determine parent-child relationships.\n\n"

        "2. PAGE NUMBER HANDLING (INTEGER REQUIRED):\n"
        "   - The target schema requires `start_page` to be an `int`. You MUST convert all page references to integers.\n"
        "   - Roman numerals (i, ii, iii, iv, xii, etc.) must be converted to their Arabic integer equivalents (1, 2, 3, 4, 12, etc.).\n"
        "   - Strip non-numeric suffixes/prefixes (e.g., '12a' -> 12, 'A-5' -> 5). Extract the leading numeric value.\n"
        "   - DO NOT apply arbitrary offsets or guess missing numbers. Use the exact printed numerical value.\n\n"

        "3. UNIQUE KEY GENERATION (`unique_key` FOR TOCEntry ONLY):\n"
        "   - Format: `{sequential_number}_{slugified_title}`\n"
        "   - Sequential Number: Zero-padded two-digit integer starting from '01' for each `TOCEntry` in reading order.\n"
        "   - Slugification Rules:\n"
        "     a. Convert title to lowercase.\n"
        "     b. Replace spaces and punctuation with underscores.\n"
        "     c. Keep ONLY lowercase letters (a-z), numbers (0-9), and underscores (_).\n"
        "     d. Collapse multiple underscores into one. Trim leading/trailing underscores.\n"
        "   - British English: Apply British spelling conventions to the slug where applicable.\n"
        "   - Example: 'Chapter 3: Agricultural Analysis' -> '03_agricultural_analysis'\n"
        "   - NOTE: Subtopics (`topics` list) do NOT receive a `unique_key`.\n\n"

        "4. TITLE CLEANING:\n"
        "   - Remove dotted leaders (e.g., '.......'), page numbers, and redundant labels (e.g., 'Chapter 1:') unless semantically critical.\n"
        "   - Preserve original capitalization or normalize to Title Case for readability.\n\n"

        "### OUTPUT FORMAT\n"
        "Return ONLY a valid JSON object strictly adhering to the `RawTOCDataset` schema.\n"
        "Do NOT include markdown formatting, code fences, explanations, or conversational text.\n"
        "Ensure all `start_page` values are integers, `topics` is always a list, and `unique_key` matches the slug pattern.\n\n"

        "### EXPECTED JSON STRUCTURE\n"
        "{\n"
        "  \"entries\": [\n"
        "    {\n"
        "      \"unique_key\": \"string (pattern: ^\\\\d{2}_[a-z0-9_]+$)\",\n"
        "      \"title\": \"string (cleaned chapter title)\",\n"
        "      \"start_page\": 0,\n"
        "      \"topics\": [\n"
        "        { \"title\": \"string (subtopic title)\", \"start_page\": 0 }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    parser = JsonOutputParser(pydantic_object=RawTOCDataset)
    
    user_content = (
        f"Parse the raw Table of Contents text from the PDF book '{pdf_path.name}'.\n"
        f"Apply the hierarchical extraction rules and output strictly valid JSON matching the RawTOCDataset schema.\n\n"
        f"RAW Table of Contents Text:\n"
        f"\"\"\"\n"
        f"{raw_toc_text}\n"
        f"\"\"\"\n\n"
        f"{parser.get_format_instructions()}"
    )

    print("\nInitialising LLM client...")
    llm = get_llm(model_name=model_name, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction),
        ("human", "{input_text}")
    ])
    
    chain = prompt | llm | parser

    print(f"Calling LLM (model override: {model_name or 'default'}) with Structured Outputs...")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def call_llm():
        return chain.invoke({"input_text": user_content})

    try:
        raw_dict = call_llm()
        raw_dataset = RawTOCDataset(**raw_dict)
        print(f"Successfully extracted {len(raw_dataset.entries)} raw TOC entries from LLM.")
    except Exception as e:
        print(f"Error validating structured output: {e}", file=sys.stderr)
        sys.exit(2)

    # Solve adjacent chapter bounds and apply offset shift
    print("Solving adjacent page boundaries and shifting indices by offset...")
    resolved_chapters = solve_chapter_boundaries(raw_dataset, actual_pages, offset)
    return resolved_chapters


def main() -> None:
    # Resolve project root dynamically
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent if current_dir.name in ("src", "book_data_extractor") else current_dir
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(
        description="Generic Table of Contents Extractor & Alignment Solver (LLM-grounded)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pdf",
        default="/home/kaizencore/learning/llm/project-loom/agri-lm/workspace/books/pdf/target_book.pdf",
        type=str,
        required=True,
        help="Path to the target PDF book file in the workspace."
    )
    parser.add_argument(
        "--actual-pages",
        type=int,
        required=True,
        help="Total printed/last-numbered page count in the book to calculate PDF offset."
    )
    parser.add_argument(
        "--toc-pages",
        type=int,
        nargs="+",
        required=True,
        help="1-indexed list of PDF page numbers that contain the Table of Contents."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override to parse the TOC text (e.g. gemini-2.5-flash)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract the raw TOC text but do not call the LLM API or save JSON."
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    # If relative path, resolve against project root
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path

    if not pdf_path.exists():
        print(f"Error: Target PDF file not found at: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        raw_toc_text = extract_raw_toc_text(pdf_path, args.toc_pages)
        print("\n=== DRY RUN COMPLETED ===")
        print("Raw Table of Contents text extracted:")
        print("=" * 50)
        print(raw_toc_text[:1000] + "\n... [truncated for display]")
        print("=" * 50)
        print("To submit this to LLM and save the resolved TOC, run without --dry-run.")
        return

    resolved_chapters = extract_and_solve_toc(
        pdf_path=pdf_path,
        actual_pages=args.actual_pages,
        toc_pages=args.toc_pages,
        model_name=args.model
    )

    # Save to JSON named after the book file
    book_name = pdf_path.stem
    output_toc_json = project_root / "data" / "processed" / f"{book_name}_toc.json"
    output_toc_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_toc_json, "w", encoding="utf-8") as f:
        json.dump(resolved_chapters, f, indent=4, ensure_ascii=False)

    print("\n🎉 DYNAMIC TABLE OF CONTENTS SOLVED!")
    print("=" * 60)
    print(f"Total resolved chapters: {len(resolved_chapters)}")
    print(f"Output saved to: {output_toc_json}")
    print("\nSample Resolved Map:")
    for key, data in list(resolved_chapters.items())[:5]:
        print(f" - {key:<25}: PDF Range {data['range']} | Title: {data['title']}")
    if len(resolved_chapters) > 5:
        print(f" ... and {len(resolved_chapters) - 5} more chapters.")
    print("=" * 60)


if __name__ == "__main__":
    main()

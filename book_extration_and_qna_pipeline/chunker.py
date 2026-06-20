"""Generic PDF Chapter Chunker.

This module reads any PDF book, parses and cleans its text contents,
and chunks it into logical crop-specific or thematic chapters based on
a dynamically resolved JSON Table of Contents (TOC) file.

The chunked text is saved under the 'data/processed/' directory, serving
as direct input for the QnA dataset generation pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from pathlib import Path
from pypdf import PdfReader



def find_project_root() -> Path:
    """Dynamically resolve the project root by searching upwards."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / ".env").exists() or (parent / "CLAUDE.md").exists():
            return parent
    return current


PROJECT_ROOT = find_project_root()


def clean_page_text(text: str, headers_to_skip: list[str] = None) -> str:
    """Clean common extraction noise from extracted PDF page text.

    This includes removing running headers, trailing printed page numbers,
    fixing hyphens at line-ends, and removing excessive whitespaces.
    """
    # Split text into lines to filter out noise
    lines = text.split("\n")
    cleaned_lines = []


    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Skip isolated running page numbers
        if re.match(r"^\d+$", stripped):
            continue
            
        # Skip dynamically provided common headers/footers
        if headers_to_skip:
            if any(header.upper() in stripped.upper() for header in headers_to_skip):
                continue
            
        cleaned_lines.append(line)

    # Join lines back with spaces
    text_content = "\n".join(cleaned_lines)


    text_content = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text_content)
    
    # Standardise spaces without flattening structural newlines
    # (replace multiple horizontal spaces with a single space)
    text_content = re.sub(r"[ \t]+", " ", text_content)

    return text_content.strip()


def extract_chapters(
    pdf_path: Path | str,
    toc_json_path: Path | str,
    output_chapters_json: Path | str,
    headers_to_skip: list[str] = None
) -> list[dict[str, any]]:
    """Extract, clean and save booklet chapters based on defined TOC ranges.

    Returns the list of processed chapter dictionaries.
    """
    pdf_path = Path(pdf_path)
    toc_json_path = Path(toc_json_path)
    output_chapters_json = Path(output_chapters_json)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF file not found at: {pdf_path}")
    if not toc_json_path.exists():
        raise FileNotFoundError(
            f"Resolved TOC JSON file not found at: {toc_json_path}.\n"
            "Please run 'src/toc_extractor.py' first to generate it."
        )

    # Load resolved TOC mapping
    print(f"Loading dynamic TOC boundaries from: {toc_json_path.name}")
    with open(toc_json_path, "r", encoding="utf-8") as f:
        chapters_toc = json.load(f)

    print(f"Reading PDF from: {pdf_path}")
    reader = PdfReader(pdf_path)
    total_pdf_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pdf_pages}")

    processed_chapters = []

    for key, info in chapters_toc.items():
        start_page, end_page = info["range"]
        title = info["title"]
        
        print(f"Processing chapter chunk: {title} (PDF Pages {start_page} to {end_page})")
        
        chapter_pages_text = []
        safe_start = max(0, start_page - 1)
        for page_idx in range(safe_start, end_page):
            if page_idx >= total_pdf_pages:
                print(f"Warning: Page index {page_idx + 1} exceeds PDF limit of {total_pdf_pages}")
                break
                
            raw_text = reader.pages[page_idx].extract_text() or ""
            cleaned_text = clean_page_text(raw_text, headers_to_skip=headers_to_skip)
            
            if cleaned_text:
                chapter_pages_text.append(cleaned_text)

        full_chapter_text = "\n\n".join(chapter_pages_text)
        
        chapter_data = {
            "key": key,
            "title": title,
            "pdf_range": [start_page, end_page],
            "text_length": len(full_chapter_text),
            "text": full_chapter_text
        }
        processed_chapters.append(chapter_data)

    # Save to book-specific output directory
    output_chapters_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_chapters_json, "w", encoding="utf-8") as f:
        json.dump(processed_chapters, f, indent=4, ensure_ascii=False)

    print(f"Successfully processed {len(processed_chapters)} chapters.")
    print(f"Book-specific output saved to: {output_chapters_json}")


    legacy_output_path = PROJECT_ROOT / "data" / "processed" / "chapters.json"
    with open(legacy_output_path, "w", encoding="utf-8") as f:
        json.dump(processed_chapters, f, indent=4, ensure_ascii=False)
    print(f"Legacy pipeline duplicate saved to: {legacy_output_path}")

    return processed_chapters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic PDF Chunker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="target_book.pdf",
        help="Path to the target PDF book file in the workspace."
    )
    parser.add_argument(
        "--toc",
        type=str,
        help="Path to the resolved TOC JSON file. Defaults to data/processed/{book_name}_toc.json."
    )
    parser.add_argument(
        "--skip-headers",
        type=str,
        nargs="+",
        default=["AGRICULTURAL MANUAL", "CROP GUIDE"],
        help="List of headers/footers to skip during text cleaning."
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    # Resolve relative paths against project root
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    book_name = pdf_path.stem

    # Resolve default TOC JSON path
    if args.toc:
        toc_json_path = Path(args.toc)
        if not toc_json_path.is_absolute():
            toc_json_path = PROJECT_ROOT / toc_json_path
    else:
        toc_json_path = PROJECT_ROOT / "data" / "processed" / f"{book_name}_toc.json"

    output_chapters_json = PROJECT_ROOT / "data" / "processed" / f"{book_name}_chapters.json"

    extract_chapters(pdf_path, toc_json_path, output_chapters_json, args.skip_headers)


if __name__ == "__main__":
    main()

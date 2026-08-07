"""TOC Detector.

This module analyzes the first ~30 pages of a PDF to detect:
1. The exact physical PDF page numbers containing the Table of Contents.
2. The highest printed page number referenced in the TOC.
3. The title of the book.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Ensure project root is in sys.path for importing local modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "book_data_extractor" else current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from book_data_extractor.llm_factory import get_llm
except ImportError:
    from llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser


class TOCDetectionResult(BaseModel):
    """Result of Table of Contents and metadata detection."""
    toc_pages: list[int] = Field(
        description="1-indexed physical PDF page numbers that contain the Table of Contents."
    )
    highest_printed_page: int = Field(
        description="The highest printed/numbered page number referenced in the Table of Contents."
    )
    title: str = Field(
        description="The title of the book, extracted from the cover or introductory pages."
    )


def detect_toc(
    pdf_path: Path,
    pages_to_scan: int = 15,
    model_name: str | None = None
) -> TOCDetectionResult:
    """Scans the beginning of a PDF and uses LangChain LLM to identify TOC information."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"[TOC Detector] Reading first {pages_to_scan} pages of {pdf_path.name}...")
    loader = OpenDataLoaderPDFLoader(
        file_path=str(pdf_path),
        format="markdown",
        image_output="off",
        table_method="cluster"
    )
    documents = loader.load()

    pages_dict = {}
    for doc in documents:
        page_num = doc.metadata.get("page", 1)
        pages_dict[page_num] = doc.page_content or ""

    total_pages = max(pages_dict.keys()) if pages_dict else 0
    scan_limit = min(pages_to_scan, total_pages)

    # Extract text from the first N pages using OpenDataLoaderPDFLoader
    pages_text = []
    for page_num in range(1, scan_limit + 1):
        text = pages_dict.get(page_num, "")
        if text.strip():
            pages_text.append(f"--- PDF Page {page_num} ---\n{text.strip()}")

    full_scan_text = "\n\n".join(pages_text)

    system_instruction = (
        "You are a Document Layout Analyst. Your task is to analyze the text extracted "
        "from the beginning of a PDF document to locate the Table of Contents (TOC), "
        "identify the physical PDF pages containing the TOC, find the highest printed page number "
        "referenced in the TOC, and determine the book's title.\n\n"
        "Guidelines:\n"
        "1. Identify the 1-indexed physical PDF page numbers where the Table of Contents (TOC) list actually is. "
        "TOC pages contain listings of chapters or sections alongside page numbers (e.g., 'Cereals .... 1-44' or 'Rice .... 1'). "
        "Pages containing actual crop recommendation content, variety lists, or pesticide names without index mapping are NOT TOC pages.\n"
        "2. Look for the highest printed/numbered page number referenced in the TOC text list. This is the page number printed next to the last topic/annexure in the TOC list. "
        "CRITICAL: Do NOT return the physical PDF page index of the TOC itself as the 'highest_printed_page'. "
        "It must be the page number printed next to the last topic in the TOC list.\n"
        "3. Extract the clean, official title of the book/document.\n"
        "4. Return ONLY a valid JSON object matching the TOCDetectionResult schema."
    )

    parser = JsonOutputParser(pydantic_object=TOCDetectionResult)
    
    user_content = (
        f"Analyze the following text from the first {scan_limit} pages of the PDF to extract Table of Contents metadata.\n\n"
        f"EXTRACTED TEXT:\n"
        f"\"\"\"\n"
        f"{full_scan_text}\n"
        f"\"\"\"\n\n"
        f"{parser.get_format_instructions()}"
    )

    # Initialize LangChain LLM
    print(f"[TOC Detector] Initialising LLM (model override: {model_name or 'default'})...")
    llm = get_llm(model_name=model_name, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction),
        ("human", "{input_text}")
    ])

    chain = prompt | llm | parser

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def call_llm():
        print(f"[TOC Detector] Querying LLM to locate TOC boundaries...")
        return chain.invoke({"input_text": user_content})

    try:
        raw_result = call_llm()
        result = TOCDetectionResult(**raw_result)
        print(f"[TOC Detector] Detected Book Title: '{result.title}'")
        print(f"[TOC Detector] Detected TOC physical PDF pages: {result.toc_pages}")
        print(f"[TOC Detector] Highest printed page in TOC: {result.highest_printed_page}")
        return result
    except Exception as e:
        print(f"[TOC Detector] Error during TOC detection: {e}", file=sys.stderr)
        raise e


def main() -> None:
    # Find project root for loading env
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent if current_dir.name == "book_data_extractor" else current_dir
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(description="Auto Detect Table of Contents boundaries from PDF.")
    parser.add_argument("--pdf", required=True, type=str, help="Path to PDF file")
    parser.add_argument("--pages-to-scan", type=int, default=15, help="Number of pages to scan at the start")
    parser.add_argument("--model", type=str, default=None, help="Model override to use")

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path

    try:
        result = detect_toc(pdf_path, args.pages_to_scan, args.model)
        print("\n--- Detection Results ---")
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"Failed to detect TOC: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

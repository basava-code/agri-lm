import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "book_data_extractor" else current_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

import fitz
from book_data_extractor.llm_factory import get_llm
from book_data_extractor.toc_detector import TOCDetectionResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

def debug_detect_toc():
    pdf_path = project_root / "target_book.pdf"
    pages_to_scan = 8
    
    doc = fitz.open(pdf_path)
    scan_limit = min(pages_to_scan, len(doc))
    
    pages_text = []
    for i in range(scan_limit):
        page_num = i + 1
        page = doc[i]
        text = page.get_text() or ""
        if text.strip():
            pages_text.append(f"--- PDF Page {page_num} ---\n{text.strip()}")

    full_scan_text = "\n\n".join(pages_text)

    # Let's inspect the system instruction
    from book_data_extractor.toc_detector import detect_toc
    print("PDF Page 3 Text:")
    print(doc[2].get_text()[:600])
    print("\nPDF Page 4 Text:")
    print(doc[3].get_text()[:600])

    print("\nInitializing LLM...")
    llm = get_llm(model_name=None, temperature=0.0)
    
    # Let's invoke the model directly with a basic prompt first to see if it parses TOC correctly
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Document Layout Analyst. Extract Table of Contents metadata from the text.\n"
            "Return a JSON object with keys: 'toc_pages' (list of physical page numbers), "
            "'highest_printed_page' (the highest printed page number listed in the TOC, e.g. 164), "
            "and 'title' (book title)."
        )),
        ("human", f"Text:\n{full_scan_text}")
    ])
    
    chain = prompt | llm
    print("\nCalling raw model...")
    response = chain.invoke({})
    print("\nRaw Model Response:")
    print(response.content)

if __name__ == "__main__":
    debug_detect_toc()

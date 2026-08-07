"""Unit Test Suite for PDF Ingestion, Dynamic TOC Alignment and Generic Chunking.

This module uses Python's standard 'unittest' framework to verify that the
boundary solver in 'src/toc_extractor.py' and the generic chunking logic in
'src/chunker.py' operate correctly, clean page text, and handle page offset alignment.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from book_data_extractor.chunker import clean_page_text, extract_chapters, find_project_root
    from book_data_extractor.toc_extractor import solve_chapter_boundaries, RawTOCDataset, TOCEntry
except ImportError:
    from chunker import clean_page_text, extract_chapters, find_project_root
    from toc_extractor import solve_chapter_boundaries, RawTOCDataset, TOCEntry


class TestGenericChunkingPipeline(unittest.TestCase):
    """Test suite covering dynamic boundary solving, text cleaning, and generic chunking."""

    def test_project_root_resolution(self) -> None:
        """Verify that the project root is found and contains key markers."""
        root = find_project_root()
        self.assertTrue(root.exists(), "Project root path must exist.")
        
        # Check that we can locate the main PDF or git structure in the resolved root
        pdf_exists = (root / "target_book.pdf").exists()
        git_exists = (root / ".git").exists()
        
        self.assertTrue(
            pdf_exists or git_exists, 
            "Resolved project root should contain repository files or markers."
        )

    def test_text_cleaning_hyphenation(self) -> None:
        """Verify that clean_page_text passes through normal words correctly."""
        raw_text = "Standard nursery transplanting techniques should be used."
        cleaned = clean_page_text(raw_text)
        
        self.assertIn("transplanting", cleaned)

    def test_text_cleaning_header_noise(self) -> None:
        """Verify that running page headers are stripped from the extracted text."""
        raw_text = "AGRICULTURAL MANUAL\nCROP GUIDE\nRice requires constant standing water."
        cleaned = clean_page_text(raw_text, headers_to_skip=["AGRICULTURAL MANUAL", "CROP GUIDE"])

        self.assertIn("Rice requires constant standing water.", cleaned)
        self.assertNotIn("AGRICULTURAL MANUAL", cleaned)
        self.assertNotIn("CROP GUIDE", cleaned)

    def test_text_cleaning_page_number_noise(self) -> None:
        """Verify that solitary page numbers are stripped from extracted text."""
        raw_text = "124\nORGANIC FARMING\nThis is printed page 124 text."
        cleaned = clean_page_text(raw_text)

        self.assertIn("ORGANIC FARMING", cleaned)
        self.assertNotIn("124", cleaned.split("\n"))

    def test_dynamic_boundary_solver(self) -> None:
        """Verify that adjacent chapter printed bounds are resolved correctly with offset shifting."""
        # Mock raw extracted entries
        mock_entries = [
            TOCEntry(unique_key="01_intro", title="Introduction", start_page=1, topics=[]),
            TOCEntry(unique_key="02_install", title="Installation Guide", start_page=20, topics=[]),
            TOCEntry(unique_key="03_advanced", title="Advanced Topics", start_page=45, topics=[])
        ]
        raw_dataset = RawTOCDataset(entries=mock_entries)
        
        actual_pages = 100
        offset = 8
        
        resolved = solve_chapter_boundaries(raw_dataset, actual_pages, offset)
        
        # We expect 3 resolved chapters
        self.assertEqual(len(resolved), 3)
        
        # Chapter 1 starts at 1, ends at 19 (since Chapter 2 starts at 20)
        # Shifted by +8 offset: PDF range should be [1+8, 19+8] = [9, 27]
        self.assertEqual(resolved["01_intro"]["range"], [9, 27])
        self.assertEqual(resolved["01_intro"]["title"], "Introduction")
        
        # Chapter 2 starts at 20, ends at 44 (since Chapter 3 starts at 45)
        # Shifted by +8 offset: PDF range should be [20+8, 44+8] = [28, 52]
        self.assertEqual(resolved["02_install"]["range"], [28, 52])
        
        # Chapter 3 starts at 45, ends at 100 (total printed pages)
        # Shifted by +8 offset: PDF range should be [45+8, 100+8] = [53, 108]
        self.assertEqual(resolved["03_advanced"]["range"], [53, 108])

    def test_generic_chunker_extraction(self) -> None:
        """Verify that chunker.py can load from dynamic JSON and extract files."""
        root = find_project_root()
        temp_dir = root / "data" / "processed"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_toc_json = temp_dir / "temp_test_toc.json"
        temp_output_json = temp_dir / "temp_test_chapters.json"
        
        # Create a mock pre-solved TOC mapping for target_book.pdf using a tiny verified slice
        # Cereals - Rice (Printed page 1 to 2 -> PDF Pages 9 to 10)
        mock_toc = {
            "test_rice_slice": {
                "range": [9, 10],
                "title": "Cereals - Rice Test Slice"
            }
        }
        
        try:
            with open(temp_toc_json, "w", encoding="utf-8") as f:
                json.dump(mock_toc, f, indent=4)
                
            # Execute the chunker
            pdf_path = root / "target_book.pdf"
            if not pdf_path.exists():
                self.skipTest(f"Sample PDF {pdf_path} not found.")
            extract_chapters(pdf_path, temp_toc_json, temp_output_json)
            
            self.assertTrue(temp_output_json.exists(), "Chunker output must exist on disk.")
            
            with open(temp_output_json, "r", encoding="utf-8") as f:
                chapters_data = json.load(f)
                
            self.assertEqual(len(chapters_data), 1)
            self.assertEqual(chapters_data[0]["key"], "test_rice_slice")
            self.assertEqual(chapters_data[0]["pdf_range"], [9, 10])
            self.assertGreater(chapters_data[0]["text_length"], 0)
            self.assertIsInstance(chapters_data[0]["text"], str)
            
        finally:
            # Clean up temporary test files
            if temp_toc_json.exists():
                temp_toc_json.unlink()
            if temp_output_json.exists():
                temp_output_json.unlink()


if __name__ == "__main__":
    unittest.main()

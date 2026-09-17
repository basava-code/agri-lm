"""Unit Test Suite for Table Repair.

Verifies that 'table_repair.py' correctly:
- unwraps single-cell image tables,
- drops chart-artifact tables (<br> soup, inconsistent columns),
- rebuilds flattened borderless tables (list records + wrapped numbers +
  displaced header fragments + summary rows) into markdown tables,
- leaves normal prose and legitimate bullet lists untouched.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from table_repair import repair_tables, reconstruct_flattened_tables, repair_markdown_blocks


class TestMarkdownBlockRepair(unittest.TestCase):

    def test_single_cell_image_table_unwrapped(self):
        text = "Before\n|![](data:image/png;base64,AAAA)|\n|---|\nAfter"
        out, stats = repair_markdown_blocks(text)
        self.assertIn("![ ](data:image/png;base64,AAAA)".replace(" ", ""), out.replace(" ", ""))
        self.assertNotIn("|---|", out)
        self.assertEqual(stats["unwrapped_images"], 1)

    def test_chart_artifact_table_dropped(self):
        soup = (
            "||0.61<br><br>36.2|5| |61.84| | |\n"
            "|---|---|---|---|---|---|\n"
            "|6.24 24.4|1|62|.14| | |"
        )
        out, stats = repair_markdown_blocks(f"Lead in.\n{soup}\nTrail.")
        self.assertNotIn("<br>", out)
        self.assertNotIn("0.61", out)
        self.assertEqual(stats["dropped_artifacts"], 1)

    def test_chart_artifact_with_leading_br_junk_line_fully_removed(self):
        soup = (
            "|0.61<br>36.25|61.84| | |\n"
            "|---|---|---|---|\n"
            "<br>0 20 40 60 80 100 120<br>2012<br>2025<br>Area (mha)<br>Strongly acidic<br>\n"
            "|---|\n"
            "Fig 4. Caption text after the artifact."
        )
        out, stats = repair_markdown_blocks(f"Prose line.\n{soup}")
        self.assertNotIn("<br>", out)
        self.assertNotIn("|---|", out)
        self.assertIn("Fig 4. Caption text", out)
        self.assertEqual(stats["dropped_artifacts"], 1)

    def test_consistent_table_kept(self):
        table = "| State | pH |\n|---|---|\n| Assam | 5.2 |"
        out, stats = repair_markdown_blocks(f"Intro\n{table}\nOutro")
        self.assertIn("| Assam | 5.2 |", out)
        self.assertEqual(stats["kept_tables"], 1)


FLATTENED = """\
S. No.
State Strongly
acidic (pH <= 4.50)
Moderately acidic (pH > 4.50 to <= 5.50)
Total Total
% of TGA
- 1 Andhra Pradesh 0.03 0.51 9.57 10.11 162.97 6.20
- 2 Arunachal Pradesh
0.17 3.14 5.05 8.36 8.37 99.88
- 3 Assam 0.15 7.15 1.20 8.49 8.49 100
India 0.61 36.25 61.84 98.71 318.37 31.00
Area (%) 0.19 11.39 19.42 31.00 100.00

Follow-up prose paragraph explaining the table contents at length with a sentence.
"""


class TestFlattenedReconstruction(unittest.TestCase):

    def test_flattened_table_rebuilt(self):
        out, rebuilt = reconstruct_flattened_tables(FLATTENED)
        self.assertEqual(rebuilt, 1)
        self.assertIn("| No. | Item | Field 1 | Field 2 | Field 3 | Field 4 | Field 5 | Field 6 |", out)
        self.assertIn("| 3 | Assam | 0.15 | 7.15 | 1.20 | 8.49 | 8.49 | 100 |", out)
        self.assertIn(
            "| 2 | Arunachal Pradesh | 0.17 | 3.14 | 5.05 | 8.36 | 8.37 | 99.88 |",
            out.replace("\n0.17", "").replace("\n", " ") if False else out,
        )
        self.assertIn("|  | India | 0.61 | 36.25 | 61.84 | 98.71 | 318.37 | 31.00 |", out)
        self.assertIn("|  | Area (%) | 0.19 | 11.39 | 19.42 | 31.00 | 100.00 |  |", out)

    def test_displaced_header_hoisted_above_table(self):
        out, _ = reconstruct_flattened_tables(FLATTENED)
        header_idx = out.index("[Recovered table header/caption fragments")
        assam_idx = out.index("| 3 | Assam")
        andhra_idx = out.index("| 1 | Andhra Pradesh")
        self.assertLess(header_idx, andhra_idx)
        self.assertLess(andhra_idx, assam_idx)

    def test_following_prose_preserved(self):
        out, _ = reconstruct_flattened_tables(FLATTENED)
        self.assertIn("Follow-up prose paragraph", out)
        self.assertNotIn("|  | Follow-up prose paragraph", out)

    def test_prose_bullets_untouched(self):
        prose = (
            "Key facts:\n"
            "- About 31 % area of the country has soil pH <= 6.50.\n"
            "- Soil pH varies from 2.80 to 10.20 with mean of 6.98.\n"
            "- 100% area in Assam has acidic soil.\n"
            "- In the last 13 years the area decreased from 6.24 to 0.61 mha."
        )
        out, rebuilt = reconstruct_flattened_tables(prose + "\n")
        self.assertEqual(rebuilt, 0)
        self.assertIn("- About 31 % area of the country", out)

    def test_mid_table_displaced_header_does_not_split_cluster(self):
        text = (
            "S. No.\n"
            "State Total\n"
            "% of TGA\n"
            "- 1 Assam 0.15 7.15 1.20 8.49 8.49 100\n"
            "- 2 Bihar 0.00 0.42 0.75 1.16 9.46 12.30\n"
            "- 3 Goa 0.02 0.34 0.01 0.37 0.37 100\n"
            "\n"
            "S. No.\n"
            "State Total\n"
            "% of TGA\n"
            "- 4 Kerala 0.00 1.64 2.20 3.85 3.88 99.14\n"
            "- 5 Punjab 0.00 0.00 0.01 0.01 5.07 0.14\n"
            "- 6 Sikkim 0.01 0.71 0.00 0.72 0.72 100\n"
            "India 0.18 10.27 4.17 14.60 27.99 52.16\n"
        )
        out, rebuilt = reconstruct_flattened_tables(text)
        self.assertEqual(rebuilt, 1)
        self.assertIn("| 3 | Goa", out)
        self.assertIn("| 6 | Sikkim", out)
        self.assertIn("|  | India | 0.18 | 10.27 | 4.17 | 14.60 | 27.99 | 52.16 |", out)
        self.assertEqual(out.count("| No. | Item"), 1)


class TestRepairTablesEndToEnd(unittest.TestCase):

    def test_composed_repair_on_mixed_text(self):
        mixed = (
            "# Chapter\n"
            "Text with figure below.\n"
            "|![](data:image/png;base64,BBBB)|\n"
            "|---|\n"
            + FLATTENED
        )
        out, stats = repair_tables(mixed)
        self.assertEqual(stats["rebuilt_flattened"], 1)
        self.assertEqual(stats["unwrapped_images"], 1)
        self.assertNotIn("<br>", out)
        self.assertIn("| 1 | Andhra Pradesh", out)


if __name__ == "__main__":
    unittest.main()

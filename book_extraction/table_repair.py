"""Table Repair Module for PDF-extracted chapter text.

Repairs the three corruption modes produced during PDF->markdown extraction:

A) Chart-artifact tables: ``table_method="cluster"`` force-fits figures/charts
   into markdown tables, producing <br> soup, split numbers ("36.2|5") and
   leaked axis labels. These blocks are dropped (their content is recovered by
   VLM figure captioning downstream).
B) Single-cell image tables: "|![](...)|\\n|---|" wrappers are unwrapped to
   bare image markdown so the captioning pipeline treats them as figures.
C) Flattened borderless tables: genuine tables without ruling borders are
   flattened into list-record text ("- 3 Assam 0.15 7.15 ..."), often with
   wrapped numeric continuation lines, a displaced column-header block, and
   trailing summary rows. This module rebuilds them into proper markdown
   tables, hoisting recovered header fragments above the table.

Also usable standalone to backfill an existing *_chapters.json:
    python table_repair.py --chapters-json "path/to/book_chapters.json"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

FLOAT_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
RECORD_START_RE = re.compile(r"^\s*[-•*]\s+(\d{1,3})[.)]?\s+(\S.*)$")
IMAGE_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
SENTENCE_BREAK_RE = re.compile(r"\.\s+[A-Z]")
LABEL_WORD_RE = re.compile(r"^[A-Za-z()%&/.+-]+$")


def _float_like(token: str) -> bool:
    return bool(FLOAT_RE.fullmatch(token))


def _numeric_payload(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    tokens = stripped.split()
    if len(tokens) < 2 or not all(_float_like(t) for t in tokens):
        return None
    return tokens


def _split_trailing_floats(tokens: list[str]) -> tuple[list[str], list[str]]:
    vals: list[str] = []
    while tokens and _float_like(tokens[-1]):
        vals.insert(0, tokens.pop())
    return tokens, vals


def _is_summary_row(line: str) -> tuple[str, list[str]] | None:
    stripped = line.strip()
    tokens = stripped.split()
    if len(tokens) < 3:
        return None
    head, tail = _split_trailing_floats(tokens[:])
    if len(tail) < 2 or not (1 <= len(head) <= 4):
        return None
    if not all(LABEL_WORD_RE.match(w) for w in head):
        return None
    return " ".join(head), tail


def _hint_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if HEADING_RE.match(stripped) or IMAGE_LINE_RE.match(stripped):
        return False
    if not any(c.isalpha() for c in stripped):
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    if len(words) > 3 and SENTENCE_BREAK_RE.search(stripped):
        return False
    return True


def _collect_leading_hints(lines: list[str], start: int) -> tuple[list[str], int]:
    """Walk backwards from a cluster start collecting contiguous header/caption
    fragment lines (displaced headers sometimes land ABOVE the first record)."""
    hints: list[str] = []
    j = start - 1
    blank_gap = 0
    while j >= 0 and len(hints) < 15:
        stripped = lines[j].strip()
        if not stripped:
            blank_gap += 1
            if blank_gap > 1:
                break
            j -= 1
            continue
        if _hint_line(lines[j]):
            hints.insert(0, stripped)
            blank_gap = 0
            j -= 1
            continue
        break
    return hints, j + 1


def _scan_flattened_cluster(lines: list[str], start: int) -> dict | None:
    """Scan forward from a record-start line. Returns cluster or None."""
    n = len(lines)
    first = RECORD_START_RE.match(lines[start])

    def parse_record(m: re.Match, at: int) -> tuple[dict, int] | None:
        serial, rest = m.group(1), m.group(2).strip()
        tokens = rest.split()
        head, vals = _split_trailing_floats(tokens)
        name = " ".join(head)
        j = at + 1
        gap = 0
        while j < n:
            stripped = lines[j].strip()
            if not stripped:
                gap += 1
                if gap > 2:
                    break
                j += 1
                continue
            payload = _numeric_payload(lines[j])
            if payload is not None:
                vals.extend(payload)
                gap = 0
                j += 1
                continue
            break
        if not name:
            return None
        return {"serial": serial, "name": name, "vals": vals}, j

    rec, next_j = parse_record(first, start)
    if rec is None:
        return None
    records: list[dict] = [rec]
    hints: list[str] = []
    footers: list[tuple[str, list[str]]] = []
    j = next_j
    blank_gap = 0

    while j < n:
        ln = lines[j]
        stripped = ln.strip()
        if not stripped:
            blank_gap += 1
            if blank_gap > 3:
                break
            j += 1
            continue
        blank_gap = 0
        m = RECORD_START_RE.match(ln)
        if m:
            parsed = parse_record(m, j)
            if parsed is None:
                break
            rec, j = parsed
            records.append(rec)
            continue
        payload = _numeric_payload(ln)
        if payload is not None and records:
            records[-1]["vals"].extend(payload)
            j += 1
            continue
        if records:
            summary = _is_summary_row(ln)
            if summary is not None:
                footers.append(summary)
                j += 1
                continue
        if len(hints) < 15 and _hint_line(ln):
            hints.append(stripped)
            j += 1
            continue
        break

    if len(records) < 3:
        return None
    widths = [len(r["vals"]) for r in records]
    modal_width = Counter(widths).most_common(1)[0][0]
    if modal_width < 2:
        return None
    leading_hints, emit_start = _collect_leading_hints(lines, start)
    seen: set[str] = set()
    all_hints = []
    for h in leading_hints + hints:
        if h not in seen:
            seen.add(h)
            all_hints.append(h)
    if len(all_hints) > 20:
        return None
    return {
        "start": start,
        "emit_start": emit_start,
        "end": j,
        "records": records,
        "hints": all_hints,
        "footers": footers,
        "width": modal_width,
    }


def _emit_cluster(cluster: dict) -> str:
    width = cluster["width"]
    parts: list[str] = []
    if cluster["hints"]:
        parts.append("[Recovered table header/caption fragments (source formatting was fragmented):]")
        parts.extend(cluster["hints"])
        parts.append("")

    header_cells = ["No.", "Item"] + [f"Field {i}" for i in range(1, width + 1)]
    parts.append("| " + " | ".join(header_cells) + " |")
    parts.append("|" + "---|" * len(header_cells))

    def fmt_row(no: str, item: str, vals: list[str]) -> str:
        cells = [no, item]
        vals = list(vals[:width]) + [""] * max(0, width - len(vals))
        cells.extend(vals)
        return "| " + " | ".join(cells) + " |"

    for rec in cluster["records"]:
        parts.append(fmt_row(rec["serial"], rec["name"], rec["vals"]))
    for label, vals in cluster["footers"]:
        parts.append(fmt_row("", label, vals))
    return "\n".join(parts)


def reconstruct_flattened_tables(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    rebuilt = 0
    while i < len(lines):
        m = RECORD_START_RE.match(lines[i])
        if m:
            candidate = _scan_flattened_cluster(lines, i)
            if candidate is not None:
                consumed_leading = candidate["start"] - candidate["emit_start"]
                if consumed_leading > 0:
                    del out[len(out) - consumed_leading:]
                out.append(_emit_cluster(candidate))
                rebuilt += 1
                i = candidate["end"]
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), rebuilt


SEPARATOR_LINE_RE = re.compile(r"^\s*\|[\s:\-|]*\|?\s*$")
BR_LINE_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _is_table_junk_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return False
    return bool(BR_LINE_RE.search(stripped)) or bool(SEPARATOR_LINE_RE.match(stripped))


def _iter_pipe_blocks(text: str) -> list[tuple[int, int, list[str]]]:
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and (
                lines[j].lstrip().startswith("|") or _is_table_junk_line(lines[j])
            ):
                j += 1
            blocks.append((i, j, lines[i:j]))
            i = j
        else:
            i += 1
    return blocks


def repair_markdown_blocks(text: str) -> tuple[str, dict]:
    lines = text.split("\n")
    stats = {"unwrapped_images": 0, "dropped_artifacts": 0, "kept_tables": 0}
    drop_spans: set[int] = set()
    replacements: dict[int, list[str]] = {}

    for start, end, block in _iter_pipe_blocks(text):
        content_rows = [
            ln for ln in block
            if not re.fullmatch(r"\s*\|[\s:\-|]*\|\s*", ln)
        ]
        if not content_rows:
            continue

        def row_cells(row: str) -> list[str]:
            core = row.strip()
            if core.startswith("|"):
                core = core[1:]
            if core.endswith("|"):
                core = core[:-1]
            return [c.strip() for c in core.split("|")]

        image_rows = [r for r in content_rows if re.fullmatch(r"\s*\|?\s*!\[[^\]]*\]\([^)]*\)\s*\|?\s*", r)]
        if len(image_rows) == len(content_rows):
            replacements[start] = [
                re.search(r"!\[[^\]]*\]\([^)]*\)", r).group(0) for r in image_rows
            ]
            stats["unwrapped_images"] += len(image_rows)
            drop_spans.update(range(start, end))
            continue

        cleaned_rows = [BR_LINE_RE.sub("; ", r) for r in content_rows]
        has_br = any("<br" in r.lower() for r in content_rows)
        counts = {len(row_cells(r)) for r in content_rows}
        empty_ratio = sum(
            1 for r in content_rows for c in row_cells(r) if c == ""
        ) / max(1, sum(len(row_cells(r)) for r in content_rows))

        if len(counts) > 1 or empty_ratio > 0.4:
            drop_spans.update(range(start, end))
            stats["dropped_artifacts"] += 1
        else:
            stats["kept_tables"] += 1
            if has_br:
                replacements[start] = cleaned_rows + [""] * (end - start - len(cleaned_rows))
                drop_spans.update(range(start, end))

    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in drop_spans:
            if i in replacements:
                out.extend(replacements[i])
            i += 1
            continue
        out.append(lines[i])
        i += 1

    def _neighbor_is_pipe(seq) -> bool:
        for x in seq:
            if x.strip():
                return x.lstrip().startswith("|")
        return False

    final: list[str] = []
    for idx, ln in enumerate(out):
        stripped = ln.strip()
        if BR_LINE_RE.search(stripped) and not stripped.startswith("|"):
            continue
        if (
            not stripped.startswith("|")
            and SEPARATOR_LINE_RE.match(ln)
            and not _neighbor_is_pipe(reversed(final))
            and not _neighbor_is_pipe(out[idx + 1:])
        ):
            continue
        final.append(ln)
    return "\n".join(final), stats


def repair_tables(text: str) -> tuple[str, dict]:
    text, block_stats = repair_markdown_blocks(text)
    text, rebuilt = reconstruct_flattened_tables(text)
    stats = {**block_stats, "rebuilt_flattened": rebuilt}
    return text, stats


def repair_chapters_json(path: Path, backup: bool = True) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        chapters = json.load(f)
    if backup:
        backup_path = path.with_suffix(".pre_table_fix.json")
        if not backup_path.exists():
            backup_path.write_text(json.dumps(chapters, ensure_ascii=False, indent=4), encoding="utf-8")
    totals = {"unwrapped_images": 0, "dropped_artifacts": 0, "kept_tables": 0, "rebuilt_flattened": 0}
    for ch in chapters:
        ch["text"], stats = repair_tables(ch["text"])
        ch["text_length"] = len(ch["text"])
        for k in totals:
            totals[k] += stats.get(k, 0)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chapters, f, indent=4, ensure_ascii=False)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair flattened/mangled tables in an extracted chapters JSON.")
    parser.add_argument("--chapters-json", required=True, help="Path to *_chapters.json produced by the chunker.")
    parser.add_argument("--no-backup", action="store_true", help="Skip writing .pre_table_fix.json backup.")
    args = parser.parse_args()
    path = Path(args.chapters_json)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    totals = repair_chapters_json(path, backup=not args.no_backup)
    print(f"[Table Repair] {json.dumps(totals)}")
    print(f"[Table Repair] Updated {path}")


if __name__ == "__main__":
    main()

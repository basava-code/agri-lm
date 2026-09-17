"""Async Local Batch Generator for CPT Corpus Creation (v2).

Key changes vs v1:
- Images are captioned ONCE per chapter by a VLM and cached to a sidecar file.
- Captions are injected as dense [FIGURE n: ...] annotations at the exact source
  position, so every downstream representation call is pure text (no context bloat).
- Per-representation generation runs through a semaphore-bounded async pool with
  tenacity retries, hard timeouts, failure isolation, and resume support.
"""
import asyncio
import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "cpt_book_pipeline" else current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

from book_extration_and_qna_pipeline.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from prompts_cpt import (
    CPT_REPRESENTATION_TYPES,
    CPT_USER_TEMPLATE,
    count_numeric_anchors,
    resolve_representation_type,
)

IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((data:image/[^;]+;base64,[^\)]+)\)")
RETRYABLE_ERRORS = (asyncio.TimeoutError, ConnectionError, OSError)

_llm_cache: dict[float, object] = {}


def prepare_image_for_model(b64_data_uri: str, max_side: int = 896) -> str:
    header, b64_data = b64_data_uri.split(",", 1)
    img_bytes = base64.b64decode(b64_data + "==")
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    resized_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{resized_b64}"


def split_segments(raw_text: str) -> list[tuple[str, str]]:
    segments = []
    last_idx = 0
    for match in IMAGE_PATTERN.finditer(raw_text):
        text_segment = raw_text[last_idx:match.start()]
        if text_segment.strip():
            segments.append(("text", text_segment))
        segments.append(("image", match.group(2).strip()))
        last_idx = match.end()
    remaining = raw_text[last_idx:]
    if remaining.strip():
        segments.append(("text", remaining))
    return segments


def interleave_captions(segments: list[tuple[str, str]], captions: dict[int, str]) -> str:
    parts = []
    fig_no = 0
    for seg_type, payload in segments:
        if seg_type == "text":
            parts.append(payload)
        else:
            desc = captions.get(fig_no, "").strip() or "visual asset present in the original book page"
            parts.append(f"\n[FIGURE {fig_no}: {desc}]\n")
            fig_no += 1
    return "".join(parts)


def load_caption_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        cache[row["chunk_key"]] = row["captions"]
                    except Exception:
                        pass
    return cache


def save_caption(cache_path: Path, chunk_key: str, captions: list[str]) -> None:
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_key": chunk_key, "captions": captions}, ensure_ascii=False) + "\n")


CAPTION_SYSTEM = (
    "You are a technical figure analyst for ICAR soil science publications. "
    "You will be given N images extracted from an agricultural reference book, in order. "
    "For EACH image write a dense 3-6 sentence technical description covering: what type of "
    "visual it is (map/chart/graph/photograph/table/diagram), exactly what data or relationship "
    "it shows, every legible number, axis label, legend entry, region name, or tabulated value, "
    "and its agronomic interpretation. Transcribe visible numbers EXACTLY. "
    'Respond with ONLY a JSON array of N strings, e.g. ["desc for image 0", "desc for image 1"]. '
    "No markdown fences, no extra keys."
)


def parse_caption_response(content: str, expected: int) -> list[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in caption response: {content[:200]}")
    parsed = json.loads(content[start:end + 1])
    if not isinstance(parsed, list) or len(parsed) != expected:
        raise ValueError(f"Caption count mismatch: expected {expected}, got {len(parsed)}")
    return [str(item).strip() for item in parsed]


async def caption_chapter_images(
    ch: dict,
    model_name: str | None,
    sem_vlm: asyncio.Semaphore,
    request_timeout: float,
    batch_size: int = 6,
) -> list[str]:
    raw_text = ch["text"]
    images = [m.group(2) for m in IMAGE_PATTERN.finditer(raw_text)]
    if not images:
        return []

    decoded: list[tuple[int, str] | tuple[int, None]] = []
    for idx, b64_uri in enumerate(images):
        try:
            decoded.append((idx, prepare_image_for_model(b64_uri)))
        except Exception as e:
            print(f"[CPT Generator] Image {idx} decode failed ({e}); caption will be left blank.", file=sys.stderr)
            decoded.append((idx, None))

    decodable = [(idx, uri) for idx, uri in decoded if uri is not None]
    captions: list[str] = [""] * len(images)
    if not decodable:
        print(f"[CPT Generator] No decodable images in '{ch['key']}'; skipping VLM captioning.", file=sys.stderr)
        return captions

    llm = _llm_cache.get(-1.0)
    if llm is None:
        llm = get_llm(model_name=model_name, temperature=0.0)
        _llm_cache[-1.0] = llm

    for i in range(0, len(decodable), batch_size):
        batch = decodable[i:i + batch_size]
        human_content = [{"type": "text", "text": f"Describe these {len(batch)} images in order."}]
        for _, uri in batch:
            human_content.append({"type": "image_url", "image_url": {"url": uri}})
        messages = [SystemMessage(content=CAPTION_SYSTEM), HumanMessage(content=human_content)]

        async def _invoke():
            return await asyncio.wait_for(asyncio.to_thread(llm.invoke, messages), timeout=request_timeout)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=4, max=30),
            reraise=True,
        ):
            with attempt:
                async with sem_vlm:
                    response = await _invoke()
        try:
            batch_captions = parse_caption_response(response.content, len(batch))
        except Exception as e:
            print(f"[CPT Generator] Caption parse failed ({e}); falling back to placeholders.", file=sys.stderr)
            batch_captions = []
        while len(batch_captions) < len(batch):
            batch_captions.append("")
        for (idx, _), cap in zip(batch, batch_captions):
            captions[idx] = cap
    return captions


def build_messages(ch: dict, rep_type: str, interleaved_text: str) -> tuple[list, float]:
    spec = CPT_REPRESENTATION_TYPES[rep_type]
    anchor_count = count_numeric_anchors(re.sub(r"\[FIGURE \d+:.*?\]", "", interleaved_text))
    anchor_target = max(int(anchor_count * 0.8), 1) if anchor_count else 1
    user_text = CPT_USER_TEMPLATE.format(
        book_title=ch.get("book_title", "Target Book"),
        chapter_title=ch["title"],
        numeric_anchor_count=anchor_count,
        numeric_anchor_target=anchor_target,
        text=interleaved_text,
    )
    messages = [SystemMessage(content=spec["system"]), HumanMessage(content=user_text)]
    return messages, spec["temperature"]


async def process_chunk(
    ch: dict,
    rep_type: str,
    interleaved_text: str,
    model_name: str | None,
    image_count: int,
    sem_gen: asyncio.Semaphore,
    results_file: Path,
    failures_file: Path,
    request_timeout: float,
) -> bool:
    async with sem_gen:
        key, title = ch["key"], ch["title"]
        chunk_id = f"{key}__{rep_type}"
        try:
            if rep_type == "raw":
                clean_text = IMAGE_PATTERN.sub("", ch["text"])
                row_data = {
                    "chunk_id": chunk_id,
                    "chapter_key": key,
                    "chapter_title": title,
                    "representation_type": rep_type,
                    "text": clean_text,
                    "image_count": image_count,
                    "source_pages": ch.get("pdf_range", []),
                }
                with open(results_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
                return True

            messages, temperature = build_messages(ch, rep_type, interleaved_text)
            llm = _llm_cache.get(temperature)
            if llm is None:
                llm = get_llm(model_name=model_name, temperature=temperature)
                _llm_cache[temperature] = llm

            async def _invoke():
                return await asyncio.wait_for(asyncio.to_thread(llm.invoke, messages), timeout=request_timeout)

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=2, min=4, max=60),
                reraise=True,
            ):
                with attempt:
                    response = await _invoke()

            result_text = response.content.strip()
            lowered = result_text.lower()
            refusal_phrases = (
                "no usable", "no comparable", "no quantitative",
                "does not contain any", "no relevant data", "not applicable",
                "cannot generate", "insufficient data",
            )
            is_no_data = (
                result_text.upper().strip(".! ") == "NO_USABLE_DATA"
                or len(result_text) < 300
                or (len(result_text) < 800 and any(p in lowered for p in refusal_phrases))
            )
            if is_no_data:
                row_data = {
                    "chunk_id": chunk_id,
                    "chapter_key": key,
                    "chapter_title": title,
                    "representation_type": rep_type,
                    "text": "",
                    "status": "skipped_unusable",
                    "image_count": image_count,
                    "source_pages": ch.get("pdf_range", []),
                }
                with open(results_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
                print(f"[CPT Generator] SKIP (unusable/empty, {len(result_text)} chars) '{rep_type}' for '{key}'")
                return True

            if not result_text:
                raise ValueError("Empty output from teacher model")

            anchor_source = count_numeric_anchors(interleaved_text)
            anchor_output = count_numeric_anchors(result_text)
            retention = anchor_output / anchor_source if anchor_source else 1.0

            row_data = {
                "chunk_id": chunk_id,
                "chapter_key": key,
                "chapter_title": title,
                "representation_type": rep_type,
                "text": result_text,
                "image_count": image_count,
                "numeric_anchor_retention": round(min(retention, 1.5), 3),
                "source_pages": ch.get("pdf_range", []),
            }
            with open(results_file, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(row_data, ensure_ascii=False) + "\n")
            print(f"[CPT Generator] OK '{rep_type}' for '{key}' (anchor retention: {retention:.0%})")
            return True

        except Exception as e:
            print(f"[CPT Generator] FAILED '{rep_type}' for '{key}': {e}", file=sys.stderr)
            with open(failures_file, "a", encoding="utf-8") as ff:
                ff.write(json.dumps({
                    "chunk_id": chunk_id,
                    "chapter_key": key,
                    "representation_type": rep_type,
                    "error": str(e)[:500],
                }, ensure_ascii=False) + "\n")
            return False


async def run_cpt_generator(
    input_chapters_path: Path,
    output_corpus_path: Path,
    model_name: str | None = None,
    representation_types: list[str] | None = None,
    include_raw_source: bool = True,
    max_concurrency: int = 8,
    dry_run: bool = False,
    vlm_max_concurrency: int = 4,
    request_timeout: float = 900.0,
) -> None:
    representation_types = representation_types or list(CPT_REPRESENTATION_TYPES.keys())
    resolved_types = []
    for rt in representation_types:
        canonical = resolve_representation_type(rt)
        if canonical not in CPT_REPRESENTATION_TYPES:
            raise ValueError(f"Unknown representation type: '{rt}'")
        resolved_types.append(canonical)
    resolved_types = list(dict.fromkeys(resolved_types))

    if not input_chapters_path.exists():
        if dry_run:
            print(f"[CPT Generator Dry Run] Pretending to load from {input_chapters_path}")
            chapters = []
        else:
            raise FileNotFoundError(f"Chunked chapters file not found at: {input_chapters_path}")
    else:
        with open(input_chapters_path, "r", encoding="utf-8") as f:
            chapters = json.load(f)

    failures_file = output_corpus_path.with_suffix(".failures.jsonl")
    caption_cache_path = output_corpus_path.with_suffix(".captions.jsonl")

    completed_ids = set()
    if output_corpus_path.exists():
        with open(output_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed_ids.add(json.loads(line)["chunk_id"])
                    except Exception:
                        pass

    types_to_run = resolved_types.copy()
    if include_raw_source and "raw" not in types_to_run:
        types_to_run = ["raw"] + types_to_run

    non_reference_chapters = []
    skipped_reference_keys = []
    for ch in chapters:
        marker = f"{ch['key']} {ch.get('title', '')}".lower()
        if re.search(r"reference|bibliograph|glossary|acknowledg", marker):
            skipped_reference_keys.append(ch["key"])
            continue
        non_reference_chapters.append(ch)
    if skipped_reference_keys:
        print(f"[CPT Generator] Auto-skipping reference sections (raw only): {skipped_reference_keys}")

    pending = []
    for ch in non_reference_chapters:
        for rep_type in types_to_run:
            if f"{ch['key']}__{rep_type}" not in completed_ids:
                pending.append((ch, rep_type))
    for key in skipped_reference_keys:
        ch = next(c for c in chapters if c["key"] == key)
        if include_raw_source and f"{ch['key']}__raw" not in completed_ids:
            pending.insert(0, (ch, "raw"))

    total_images = sum(len(IMAGE_PATTERN.findall(ch["text"])) for ch in non_reference_chapters)
    print(f"[CPT Generator] Chapters: {len(non_reference_chapters)} (+{len(skipped_reference_keys)} reference-only) | Types/chapter: {len(types_to_run)} | Total figures: {total_images}")
    print(f"[CPT Generator] Completed: {len(completed_ids)} | Pending: {len(pending)}")

    if dry_run:
        print("\n=== DRY RUN COMPLETED ===")
        return

    if not pending:
        print("[CPT Generator] All chunks have already been processed!")
        return

    pending_keys = list(dict.fromkeys(ch["key"] for ch, _ in pending))
    chapters_by_key = {ch["key"]: ch for ch in chapters}
    caption_cache = load_caption_cache(caption_cache_path)
    sem_vlm = asyncio.Semaphore(vlm_max_concurrency)

    print(f"[CPT Generator] Phase 1: VLM figure annotation for {len(pending_keys)} chapters...")
    phase1_start = time.time()
    interleaved_by_key: dict[str, str] = {}
    image_counts: dict[str, int] = {}

    for key in pending_keys:
        ch = chapters_by_key[key]
        n_images = len(IMAGE_PATTERN.findall(ch["text"]))
        image_counts[key] = n_images
        cached = caption_cache.get(key)
        if cached is not None and len(cached) == n_images:
            captions_list = cached
        else:
            captions_list = await caption_chapter_images(ch, model_name, sem_vlm, request_timeout)
            save_caption(caption_cache_path, key, captions_list)
        interleaved_by_key[key] = interleave_captions(split_segments(ch["text"]), {i: c for i, c in enumerate(captions_list)})
        print(f"[CPT Generator] Annotated '{key}' ({n_images} figures, {len(interleaved_by_key[key])} chars)")

    print(f"[CPT Generator] Phase 1 done in {time.time() - phase1_start:.1f}s. Phase 2: generation...")

    sem_gen = asyncio.Semaphore(max_concurrency)
    tasks = [
        process_chunk(
            ch, rep_type, interleaved_by_key[ch["key"]], model_name,
            image_counts[ch["key"]], sem_gen, output_corpus_path, failures_file, request_timeout,
        )
        for ch, rep_type in pending
    ]
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)

    succeeded = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    crashed = [r for r in results if isinstance(r, Exception)]
    for exc in crashed:
        print(f"[CPT Generator] Unexpected crash: {exc!r}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\n[CPT Generator] Completed in {elapsed:.1f}s | OK: {succeeded} | Failed: {failed} | Crashed: {len(crashed)}")
    print(f"[CPT Generator] Corpus: {output_corpus_path}")
    if failed or crashed:
        print(f"[CPT Generator] Failures logged to: {failures_file} (re-run to retry)")

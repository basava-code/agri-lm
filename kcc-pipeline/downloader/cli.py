"""
Optimized resumable Data.gov.in KCC dataset downloader.

Important:
- Concurrency stays at 1 by default because the API/load balancer returns 429
  when concurrency is increased.
- Every API page is saved immediately as a raw CSV part.
- CSV rows are counted with Python's CSV parser, not newline counting.
- API offsets advance by the requested API limit, never by physical newline
  counts.
- Pages are not converted through pandas on the hot path.
- A 100,000-row batch is combined only once, after all its pages are durable.
- checkpoint.json is updated after every durable page.
- Existing v1/v2 checkpoints are accepted, so an existing download can resume.

Required .env:
    DATA_GOV_API_KEY=your_api_key
    DATASET_ID=your_dataset_id

Optional .env:
    PAGE_LIMIT=1000
    BATCH_SIZE=100000
    CONCURRENT_REQUESTS=1
"""

import asyncio
import csv
import io
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


# Configuration
load_dotenv()

DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
DATASET_ID = os.getenv("DATASET_ID")

if not DATA_GOV_API_KEY:
    raise RuntimeError("DATA_GOV_API_KEY is missing from .env or environment.")

if not DATASET_ID:
    raise RuntimeError("DATASET_ID is missing from .env or environment.")

API_URL = f"https://api.data.gov.in/resource/{DATASET_ID}"

PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100000"))
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", "1"))

MAX_RETRIES = 5
BACKOFF_BASE_DELAY = 2.0
NETWORK_TIMEOUT = 90.0

ROOT_DIR = Path("kcc-dataset")
DATA_DIR = ROOT_DIR / "data"
TEMP_DIR = DATA_DIR / ".tmp"
CHECKPOINT_FILE = ROOT_DIR / "checkpoint.json"
LOG_FILE = ROOT_DIR / "downloader.log"

ROOT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Logging
logger = logging.getLogger("KCCDownloader")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Checkpoint
def default_checkpoint() -> dict[str, Any]:
    return {
        "version": 3,
        "status": "not_started",
        "dataset_id": DATASET_ID,
        "batch_size": BATCH_SIZE,
        "page_limit": PAGE_LIMIT,
        "concurrent_requests": CONCURRENT_REQUESTS,
        "completed_batches": 0,
        "downloaded_rows": 0,
        "next_offset": 0,
        "last_completed_batch": None,
        "last_completed_range": None,
        "current_batch": None,
        "current_batch_start": None,
        "current_batch_rows": 0,
        "current_batch_pages": 0,
        "updated_at": None,
    }


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    payload = (
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    atomic_write_bytes(path, payload)


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    checkpoint["updated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    atomic_write_json(CHECKPOINT_FILE, checkpoint)


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_FILE.exists():
        checkpoint = default_checkpoint()
        save_checkpoint(checkpoint)
        logger.info("Created new checkpoint.json.")
        return checkpoint

    try:
        with CHECKPOINT_FILE.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"checkpoint.json is corrupted: {exc}. "
            "Inspect or restore it before continuing."
        ) from exc

    defaults = default_checkpoint()
    for key, value in defaults.items():
        checkpoint.setdefault(key, value)

    if checkpoint.get("dataset_id") != DATASET_ID:
        raise RuntimeError(
            "checkpoint.json belongs to a different DATASET_ID."
        )

    return checkpoint


# CSV helpers
def count_csv_records(raw_csv: str) -> int:
    """
    Count real CSV records, excluding the header.

    This handles quoted fields containing embedded newlines correctly.
    """
    if not raw_csv.strip():
        return 0

    reader = csv.reader(io.StringIO(raw_csv, newline=""))

    try:
        next(reader)
    except StopIteration:
        return 0

    return sum(1 for row in reader if row)


def write_raw_page(path: Path, raw_csv: str) -> None:
    """
    Persist the API response without pandas parsing/re-serialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    data = raw_csv.encode("utf-8")

    with temp_path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def page_sort_key(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return 0


def remove_csv_header(raw_csv: str) -> str:
    """
    Remove exactly one CSV record (the header), while preserving the remaining
    CSV text, including embedded newlines.

    This is used only when a 100k batch is finalized.
    """
    if not raw_csv:
        return ""

    reader = csv.reader(io.StringIO(raw_csv, newline=""))

    try:
        next(reader)
    except StopIteration:
        return ""

    # csv.reader.line_num tells us how many physical lines made up the header.
    header_lines = reader.line_num

    physical_lines = raw_csv.splitlines(keepends=True)

    return "".join(physical_lines[header_lines:])


# Downloader
class ResumableDownloader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        checkpoint: dict[str, Any],
    ):
        self.client = client
        self.checkpoint = checkpoint
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    @staticmethod
    def headers() -> dict[str, str]:
        return {
            "accept": "text/csv",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

    def batch_temp_dir(self, batch_number: int) -> Path:
        return TEMP_DIR / f"batch_{batch_number:06d}"

    def page_path(
        self,
        batch_number: int,
        page_number: int,
    ) -> Path:
        return (
            self.batch_temp_dir(batch_number)
            / f"page_{page_number:06d}.csv"
        )

    def final_batch_path(
        self,
        batch_number: int,
        start: int,
        end: int,
    ) -> Path:
        return DATA_DIR / (
            f"Batch {batch_number} Range ({start} - {end}).csv"
        )

    async def fetch_page(
        self,
        offset: int,
        limit: int,
    ) -> tuple[str, int]:
        params = {
            "api-key": DATA_GOV_API_KEY,
            "format": "csv",
            "offset": offset,
            "limit": limit,
        }

        async with self.semaphore:
            for attempt in range(1, MAX_RETRIES + 1):
                started = time.perf_counter()

                try:
                    logger.info(
                        "FETCH offset=%s limit=%s attempt=%s/%s",
                        offset,
                        limit,
                        attempt,
                        MAX_RETRIES,
                    )

                    response = await self.client.get(
                        API_URL,
                        params=params,
                        headers=self.headers(),
                    )

                    elapsed = time.perf_counter() - started

                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")

                        logger.warning(
                            "HTTP 429 at offset=%s after %.2fs "
                            "(Retry-After=%s).",
                            offset,
                            elapsed,
                            retry_after or "not provided",
                        )

                        if attempt == MAX_RETRIES:
                            response.raise_for_status()

                        if retry_after:
                            try:
                                delay = max(
                                    float(retry_after),
                                    BACKOFF_BASE_DELAY,
                                )
                            except ValueError:
                                delay = BACKOFF_BASE_DELAY * (
                                    2 ** (attempt - 1)
                                )
                        else:
                            delay = BACKOFF_BASE_DELAY * (
                                2 ** (attempt - 1)
                            )

                        delay += time.time() % 1.0
                        logger.info(
                            "429 BACKOFF sleep=%.2fs",
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if response.status_code >= 500:
                        logger.warning(
                            "HTTP %s at offset=%s after %.2fs.",
                            response.status_code,
                            offset,
                            elapsed,
                        )
                        response.raise_for_status()

                    response.raise_for_status()

                    raw_csv = response.text
                    actual_rows = count_csv_records(raw_csv)

                    logger.info(
                        "RESPONSE offset=%s rows=%s time=%.2fs",
                        offset,
                        actual_rows,
                        elapsed,
                    )

                    return raw_csv, actual_rows

                except (
                    httpx.HTTPStatusError,
                    httpx.RequestError,
                    httpx.TimeoutException,
                ) as exc:
                    elapsed = time.perf_counter() - started

                    logger.warning(
                        "REQUEST ERROR offset=%s time=%.2fs: %s",
                        offset,
                        elapsed,
                        exc,
                    )

                    if attempt == MAX_RETRIES:
                        raise

                    delay = (
                        BACKOFF_BASE_DELAY * (2 ** (attempt - 1))
                    ) + (time.time() % 1.0)

                    logger.info(
                        "RETRY offset=%s sleep=%.2fs",
                        offset,
                        delay,
                    )

                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Failed to fetch offset {offset} after retries."
        )

    def discover_page_parts(
        self,
        batch_number: int,
    ) -> list[Path]:
        directory = self.batch_temp_dir(batch_number)

        if not directory.exists():
            return []

        return sorted(
            directory.glob("page_*.csv"),
            key=page_sort_key,
        )

    def remove_completed_temp_batches(self) -> None:
        completed = int(
            self.checkpoint.get("completed_batches", 0)
        )

        for directory in TEMP_DIR.glob("batch_*"):
            try:
                batch_number = int(directory.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            if batch_number <= completed:
                shutil.rmtree(directory, ignore_errors=True)

    def combine_raw_pages(
        self,
        batch_number: int,
        start_offset: int,
        row_count: int,
    ) -> Path:
        """
        Combine the raw page files.

        This avoids pandas completely. Only the first page's header is kept;
        all following page bodies are copied as CSV text.

        We verify the page counts during finalization before committing the
        final batch.
        """
        parts = self.discover_page_parts(batch_number)

        if not parts:
            raise RuntimeError(
                f"No page files found for Batch {batch_number}."
            )

        expected_end = start_offset + row_count
        final_path = self.final_batch_path(
            batch_number,
            start_offset,
            expected_end,
        )
        temp_final = TEMP_DIR / (
            f"batch_{batch_number:06d}.final.csv.tmp"
        )

        logger.info(
            "COMBINE batch=%s pages=%s rows=%s",
            batch_number,
            len(parts),
            row_count,
        )

        verified_rows = 0

        with temp_final.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as out:
            for index, part in enumerate(parts):
                raw = part.read_text(
                    encoding="utf-8",
                )

                page_rows = count_csv_records(raw)
                verified_rows += page_rows

                if index == 0:
                    out.write(raw)
                else:
                    out.write(remove_csv_header(raw))

            out.flush()
            os.fsync(out.fileno())

        if verified_rows != row_count:
            temp_final.unlink(missing_ok=True)

            raise RuntimeError(
                f"Batch {batch_number} verification failed: "
                f"checkpoint rows={row_count}, "
                f"page rows={verified_rows}."
            )

        os.replace(temp_final, final_path)

        logger.info(
            "FINAL CSV SAVED batch=%s rows=%s file=%s",
            batch_number,
            verified_rows,
            final_path.name,
        )

        return final_path

    async def download_batch(
        self,
        batch_number: int,
        start_offset: int,
    ) -> tuple[int, bool]:
        batch_start = start_offset
        current_batch = self.checkpoint.get("current_batch")

        if current_batch == batch_number:
            current_rows = int(
                self.checkpoint.get("current_batch_rows", 0)
            )
            current_offset = int(
                self.checkpoint.get("next_offset", batch_start)
            )
            page_number = int(
                self.checkpoint.get("current_batch_pages", 0)
            )
        else:
            current_rows = 0
            current_offset = batch_start
            page_number = 0

            self.checkpoint["current_batch"] = batch_number
            self.checkpoint["current_batch_start"] = batch_start
            self.checkpoint["current_batch_rows"] = 0
            self.checkpoint["current_batch_pages"] = 0
            self.checkpoint["next_offset"] = batch_start
            save_checkpoint(self.checkpoint)

        logger.info("============================================================")
        logger.info(
            "BATCH %s START/RESUME range=(%s - %s)",
            batch_number,
            batch_start,
            batch_start + BATCH_SIZE,
        )
        logger.info(
            "PROGRESS rows=%s/%s next_offset=%s page=%s",
            current_rows,
            BATCH_SIZE,
            current_offset,
            page_number,
        )

        end_of_dataset = False

        while current_rows < BATCH_SIZE:
            remaining = BATCH_SIZE - current_rows
            request_limit = min(PAGE_LIMIT, remaining)

            raw_csv, actual_rows = await self.fetch_page(
                offset=current_offset,
                limit=request_limit,
            )

            if actual_rows == 0:
                end_of_dataset = True
                break

            # A normal response should contain <= request_limit records.
            # If not, keep the safety behavior for the exceptional case.
            if actual_rows > remaining:
                raise RuntimeError(
                    f"API returned {actual_rows} records while only "
                    f"{remaining} were needed for Batch {batch_number}."
                )

            page_path = self.page_path(
                batch_number,
                page_number,
            )

            write_raw_page(
                page_path,
                raw_csv,
            )

            logger.info(
                "PAGE SAVED batch=%s page=%s offset=%s rows=%s",
                batch_number,
                page_number,
                current_offset,
                actual_rows,
            )

            current_rows += actual_rows

            # API pagination is based on the requested limit.
            #
            # Do NOT use newline counts here.
            current_offset += request_limit
            page_number += 1

            self.checkpoint["version"] = 3
            self.checkpoint["status"] = "running"
            self.checkpoint["current_batch"] = batch_number
            self.checkpoint["current_batch_start"] = batch_start
            self.checkpoint["current_batch_rows"] = current_rows
            self.checkpoint["current_batch_pages"] = page_number
            self.checkpoint["next_offset"] = current_offset

            save_checkpoint(self.checkpoint)

            logger.info(
                "PAGE CHECKPOINTED rows=%s/%s next_offset=%s",
                current_rows,
                BATCH_SIZE,
                current_offset,
            )

            if actual_rows < request_limit:
                end_of_dataset = True
                logger.info(
                    "DATASET END detected: returned %s < requested %s.",
                    actual_rows,
                    request_limit,
                )
                break

        if current_rows == 0:
            return 0, True

        final_path = self.combine_raw_pages(
            batch_number=batch_number,
            start_offset=batch_start,
            row_count=current_rows,
        )

        self.checkpoint["version"] = 3
        self.checkpoint["completed_batches"] = batch_number
        self.checkpoint["downloaded_rows"] = (
            int(self.checkpoint.get("downloaded_rows", 0))
            + current_rows
        )
        self.checkpoint["next_offset"] = current_offset

        self.checkpoint["last_completed_batch"] = batch_number
        self.checkpoint["last_completed_range"] = {
            "start": batch_start,
            "end": batch_start + current_rows,
            "rows": current_rows,
            "file": str(final_path),
        }

        self.checkpoint["current_batch"] = None
        self.checkpoint["current_batch_start"] = None
        self.checkpoint["current_batch_rows"] = 0
        self.checkpoint["current_batch_pages"] = 0

        save_checkpoint(self.checkpoint)

        logger.info(
            "BATCH COMMITTED batch=%s range=(%s - %s) rows=%s",
            batch_number,
            batch_start,
            batch_start + current_rows,
            current_rows,
        )

        shutil.rmtree(
            self.batch_temp_dir(batch_number),
            ignore_errors=True,
        )

        logger.info(
            "TEMP PAGES REMOVED batch=%s",
            batch_number,
        )

        return (
            current_rows,
            end_of_dataset or current_rows < BATCH_SIZE,
        )

    async def run(self) -> None:
        self.remove_completed_temp_batches()

        completed_batches = int(
            self.checkpoint.get("completed_batches", 0)
        )
        offset = int(
            self.checkpoint.get("next_offset", 0)
        )

        current_batch = self.checkpoint.get("current_batch")

        if current_batch is not None:
            batch_number = int(current_batch)
        else:
            batch_number = completed_batches + 1

        self.checkpoint["status"] = "running"
        save_checkpoint(self.checkpoint)

        logger.info("============================================================")
        logger.info("KCC DATASET DOWNLOADER")
        logger.info("Dataset ID: %s", DATASET_ID)
        logger.info("Page limit: %s", PAGE_LIMIT)
        logger.info("Batch size: %s", BATCH_SIZE)
        logger.info(
            "Concurrency: %s (intentionally conservative)",
            CONCURRENT_REQUESTS,
        )
        logger.info("Starting offset: %s", offset)
        logger.info("============================================================")

        while True:
            rows, is_final = await self.download_batch(
                batch_number=batch_number,
                start_offset=offset,
            )

            if rows == 0 or is_final:
                self.checkpoint["status"] = "completed"
                save_checkpoint(self.checkpoint)

                logger.info("============================================================")
                logger.info("DATASET DOWNLOAD COMPLETED")
                logger.info(
                    "Total rows: %s",
                    self.checkpoint["downloaded_rows"],
                )
                logger.info(
                    "Completed batches: %s",
                    self.checkpoint["completed_batches"],
                )
                logger.info("============================================================")
                return

            offset = int(
                self.checkpoint["next_offset"]
            )
            batch_number += 1


# Main
async def main() -> None:
    checkpoint = load_checkpoint()

    logger.info("------------------------------------------------------------")
    logger.info("KCC DATASET CLI START")
    logger.info(
        "Completed rows: %s",
        checkpoint.get("downloaded_rows", 0),
    )
    logger.info(
        "Next offset: %s",
        checkpoint.get("next_offset", 0),
    )
    logger.info(
        "Completed batches: %s",
        checkpoint.get("completed_batches", 0),
    )

    if checkpoint.get("current_batch") is not None:
        logger.info(
            "Resuming Batch %s: %s/%s rows",
            checkpoint.get("current_batch"),
            checkpoint.get("current_batch_rows", 0),
            BATCH_SIZE,
        )

    logger.info("------------------------------------------------------------")

    # One long-lived AsyncClient means TCP/TLS connections are reused.
    limits = httpx.Limits(
        max_connections=max(2, CONCURRENT_REQUESTS + 1),
        max_keepalive_connections=max(1, CONCURRENT_REQUESTS),
        keepalive_expiry=30.0,
    )

    timeout = httpx.Timeout(
        connect=20.0,
        read=NETWORK_TIMEOUT,
        write=20.0,
        pool=20.0,
    )

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        downloader = ResumableDownloader(
            client=client,
            checkpoint=checkpoint,
        )

        try:
            await downloader.run()

        except KeyboardInterrupt:
            logger.warning("Downloader interrupted by user.")
            checkpoint["status"] = "interrupted"
            save_checkpoint(checkpoint)
            raise

        except asyncio.CancelledError:
            logger.warning("Downloader cancelled.")
            checkpoint["status"] = "interrupted"
            save_checkpoint(checkpoint)
            raise

        except Exception:
            logger.exception(
                "FATAL: Downloader stopped unexpectedly."
            )
            checkpoint["status"] = "error"
            save_checkpoint(checkpoint)
            raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(
            "\nDownloader stopped. "
            "Run `python cli.py` again to resume."
        )
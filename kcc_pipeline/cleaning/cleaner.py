"""
ETL Data Cleaning and Transformation Pipeline

This module provides a production-grade Command Line Interface (CLI) to process,
clean, transform, and deduplicate large batches of CSV files. Designed for 
memory-efficiency, it processes files in chunks and maintains a persistent state
to gracefully handle interruptions and resume progress.

Usage Example:
    python etl_cleaner.py --input-dir data/raw --output-dir data/processed --chunk-size 50000

Arguments:
    --input-dir    : Directory containing raw batch CSV files (default: data/raw)
    --output-dir   : Directory to save cleaned CSV files (default: data/processed)
    --chunk-size   : Number of rows per processing chunk (default: 50000)
    --reset-state  : Flag to delete the checkpoint and restart from scratch
    -v, --verbose  : Flag to enable debug-level logging
"""

import os
import sys
import re
import json
import signal
import logging
import argparse
import hashlib
import unicodedata
from pathlib import Path
from typing import Set, Dict, Any, List
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
from tqdm import tqdm


# constants and required schema
REQUIRED_COLS = [
    "BlockName", "Category", "CreatedOn", "Crop", "DistrictName",
    "KccAns", "QueryText", "QueryType", "Sector", "StateName", "day", "month", "year"
]

OUTPUT_COLS = [
    "Id", "Question", "Answer", "Crop", "Category", "Sector",
    "QueryType", "Date & Time", "Block", "District", "State"
]

SUPPORTED_ENCODINGS = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
DEFAULT_CHUNK_SIZE = 50000

LOGS_DIR = Path("logs")
STATE_DIR = Path("state")
LOG_FILE = LOGS_DIR / "cleaner.log"
CHECKPOINT_FILE = STATE_DIR / "checkpoint.json"

# precompiled regex for text cleanup and sentence casing
WHITESPACE_REGEX = re.compile(r'[\r\n\t]+|\s{2,}')
SENTENCE_CASE_REGEX = re.compile(r'(?:^|[.!?]\s+)([a-z])')


# configure dual rotating file and console logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("ETL_Cleaner")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    # 10mb file limit with 5 backups
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # stdout stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(console_handler)

    return logger


# persistent state checkpoint manager
class CheckpointManager:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.state: Dict[str, Any] = {
            "completed_files": {},
            "in_progress": None,
            "next_id": 1,
            "seen_question_hashes": []
        }
        self.seen_hashes: Set[int] = set()
        self._load()

    def _load(self) -> None:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
                self.seen_hashes = set(self.state.get("seen_question_hashes", []))
            except Exception as e:
                logging.getLogger("ETL_Cleaner").warning(
                    f"Corrupt checkpoint found. Resetting state. Error: {e}"
                )

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.state["seen_question_hashes"] = list(self.seen_hashes)
        
        # atomic write using temp replacement
        temp_file = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        temp_file.replace(self.checkpoint_path)

    def is_file_completed(self, filename: str) -> bool:
        return filename in self.state["completed_files"]

    def mark_file_started(self, filename: str) -> None:
        self.state["in_progress"] = {
            "filename": filename,
            "chunks_processed": 0,
            "rows_written": 0
        }
        self.save()

    def mark_chunk_processed(self, filename: str, rows_written: int) -> None:
        if self.state["in_progress"] and self.state["in_progress"]["filename"] == filename:
            self.state["in_progress"]["chunks_processed"] += 1
            self.state["in_progress"]["rows_written"] += rows_written
            self.save()

    def mark_file_completed(self, filename: str, total_rows_written: int) -> None:
        self.state["completed_files"][filename] = {
            "total_rows_written": total_rows_written
        }
        self.state["in_progress"] = None
        self.save()

    def get_next_id(self) -> int:
        return self.state.get("next_id", 1)

    def increment_id(self, count: int) -> None:
        self.state["next_id"] = self.get_next_id() + count


# optimized single-pass string cleaning routine
def clean_text(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    text = str(val).strip()
    if not text:
        return ""
    
    # collapse whitespace and tabs
    text = WHITESPACE_REGEX.sub(' ', text).strip()
    
    # normalize unicode to nfc and apply sentence casing
    text = unicodedata.normalize('NFC', text).lower()
    return SENTENCE_CASE_REGEX.sub(lambda m: m.group(0).upper(), text)


# apply single-pass cleaning on a series
def clean_series(s: pd.Series) -> pd.Series:
    return pd.Series([clean_text(x) for x in s.values], index=s.index)


# fast character iteration language check without regex allocations
def is_english(text: Any) -> bool:
    if not text:
        return False
    text_str = str(text)
    total_letters = 0
    ascii_letters = 0
    
    for ch in text_str:
        if ch.isalpha():
            total_letters += 1
            if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
                ascii_letters += 1
                
    if total_letters == 0:
        return False
    return (ascii_letters / total_letters) >= 0.8


# vectorized date parsing with fallback
def parse_dates(df: pd.DataFrame) -> pd.Series:
    dt_created = pd.to_datetime(df['CreatedOn'], errors='coerce', dayfirst=True, format='mixed')

    dmy_df = pd.DataFrame({
        'year': pd.to_numeric(df['year'], errors='coerce'),
        'month': pd.to_numeric(df['month'], errors='coerce'),
        'day': pd.to_numeric(df['day'], errors='coerce')
    })
    dt_dmy = pd.to_datetime(dmy_df, errors='coerce')

    dt_final = dt_created.combine_first(dt_dmy)
    return dt_final.dt.strftime('%Y-%m-%d %H:%M:%S').fillna("")


# 64-bit integer hash for low memory deduplication tracking
def hash_text(text: str) -> int:
    return int.from_bytes(hashlib.md5(text.encode('utf-8')).digest()[:8], 'little')


# core chunked etl pipeline
class CSVPipeline:
    def __init__(self, checkpoint_mgr: CheckpointManager, logger: logging.Logger, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.checkpoint = checkpoint_mgr
        self.logger = logger
        self.chunk_size = chunk_size
        self._shutdown_requested = False

        # handle interruption signals gracefully
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        self.logger.warning("Cancellation received. Stopping after current chunk to preserve state...")
        self._shutdown_requested = True

    def detect_encoding(self, file_path: Path) -> str:
        for enc in SUPPORTED_ENCODINGS:
            try:
                header = pd.read_csv(file_path, nrows=0, encoding=enc)
                missing = [c for c in REQUIRED_COLS if c not in header.columns]
                if missing:
                    raise ValueError(f"Missing required columns: {', '.join(missing)}")
                return enc
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                raise ValueError(f"File {file_path.name} is empty.")
        raise ValueError(f"Unable to parse {file_path.name} with supported encodings.")

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        # clean text columns
        for col in REQUIRED_COLS:
            chunk[col] = clean_series(chunk[col])

        # filter non-english answers
        is_eng_mask = [is_english(x) for x in chunk["KccAns"].values]
        chunk = chunk[is_eng_mask].copy()
        if chunk.empty:
            return pd.DataFrame(columns=OUTPUT_COLS)

        # deduplicate questions using 64-bit hash tracking
        unique_mask = []
        for q_text in chunk["QueryText"].values:
            h = hash_text(q_text)
            if h not in self.checkpoint.seen_hashes:
                self.checkpoint.seen_hashes.add(h)
                unique_mask.append(True)
            else:
                unique_mask.append(False)

        chunk = chunk[unique_mask].copy()
        if chunk.empty:
            return pd.DataFrame(columns=OUTPUT_COLS)

        # construct output columns with individual location aliases
        current_id = self.checkpoint.get_next_id()
        num_rows = len(chunk)
        
        df_out = pd.DataFrame({
            "Id": np.arange(current_id, current_id + num_rows),
            "Question": chunk["QueryText"].values,
            "Answer": chunk["KccAns"].values,
            "Crop": chunk["Crop"].values,
            "Category": chunk["Category"].values,
            "Sector": chunk["Sector"].values,
            "QueryType": chunk["QueryType"].values,
            "Date & Time": parse_dates(chunk).values,
            "Block": chunk["BlockName"].values,
            "District": chunk["DistrictName"].values,
            "State": chunk["StateName"].values
        })

        self.checkpoint.increment_id(num_rows)
        return df_out[OUTPUT_COLS]

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        filename = input_path.name

        if self.checkpoint.is_file_completed(filename):
            self.logger.info(f"Skipping already completed batch: {filename}")
            return True

        self.logger.info(f"Processing batch: {filename}")
        encoding = self.detect_encoding(input_path)

        # remove previous partial temporary file if exists
        temp_output_path = output_path.with_suffix(".tmp")
        if temp_output_path.exists():
            temp_output_path.unlink()

        self.checkpoint.mark_file_started(filename)
        total_rows_written = 0

        try:
            reader = pd.read_csv(
                input_path,
                encoding=encoding,
                usecols=REQUIRED_COLS,
                dtype=str,
                chunksize=self.chunk_size,
                on_bad_lines='skip'
            )

            is_first_chunk = True
            with tqdm(desc=f"Batch: {filename}", unit="chunk") as pbar:
                for chunk in reader:
                    transformed_df = self.transform_chunk(chunk)
                    
                    if not transformed_df.empty:
                        transformed_df.to_csv(
                            temp_output_path,
                            mode='w' if is_first_chunk else 'a',
                            header=is_first_chunk,
                            index=False,
                            encoding='utf-8'
                        )
                        is_first_chunk = False
                        rows_written = len(transformed_df)
                        total_rows_written += rows_written
                    else:
                        rows_written = 0

                    self.checkpoint.mark_chunk_processed(filename, rows_written)
                    pbar.update(1)

                    if self._shutdown_requested:
                        self.logger.warning("Pipeline paused cleanly.")
                        return False

            # generate empty csv with headers if all rows were filtered out
            if is_first_chunk:
                pd.DataFrame(columns=OUTPUT_COLS).to_csv(temp_output_path, index=False, encoding='utf-8')

            # commit completed batch atomically
            temp_output_path.replace(output_path)
            self.checkpoint.mark_file_completed(filename, total_rows_written)
            self.logger.info(f"Completed {filename} ({total_rows_written} clean rows).")
            return True

        except Exception as e:
            self.logger.exception(f"Error processing {filename}: {e}")
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise


# cli argument parser and runner
def main():
    parser = argparse.ArgumentParser(
        description="Production CLI to clean, transform, and deduplicate CSV batches."
    )
    parser.add_argument(
        "--input-dir", type=str, default="data/raw",
        help="Directory containing raw batch CSV files."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Directory where cleaned CSV files are stored."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Number of rows per processing chunk (default: {DEFAULT_CHUNK_SIZE})."
    )
    parser.add_argument(
        "--reset-state", action="store_true",
        help="Reset checkpoints and restart cleaning from scratch."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug log output."
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset_state and CHECKPOINT_FILE.exists():
        logger.warning("Resetting state checkpoint.")
        CHECKPOINT_FILE.unlink()

    checkpoint_mgr = CheckpointManager(CHECKPOINT_FILE)
    pipeline = CSVPipeline(checkpoint_mgr, logger, chunk_size=args.chunk_size)

    # find all csv batch files
    csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return

    logger.info(f"Discovered {len(csv_files)} batch CSV files.")

    for file_path in csv_files:
        out_file_path = output_dir / f"cleaned_{file_path.name}"
        try:
            success = pipeline.process_file(file_path, out_file_path)
            if not success:
                logger.info("Process stopped. Re-run command to resume.")
                sys.exit(0)
        except Exception as e:
            logger.error(f"Fatal error while processing {file_path.name}: {e}")
            sys.exit(1)

    logger.info("All CSV batches processed successfully.")


if __name__ == "__main__":
    main()
"""
dataset_normalizer.py

A production-grade dataset normalization engine for the Semantic Topic Discovery Pipeline.
Responsible for cleaning, normalizing, and standardizing conversational agricultural 
queries into semantic representations suitable for embedding generation (e.g., BGE-M3).

Architecture:
- Built on Polars for blazing-fast, memory-efficient, lazy-evaluated transformations.
- Streams large datasets using PyArrow ParquetWriter and Polars batching.
- Supports checkpointing for resumability on massive datasets.
- 100% Type-hinted, adheres to SOLID, DRY, and PEP8.

Usage:
    python dataset_normalizer.py \
        --input data/raw/questions.csv \
        --output data/cleaned \
        --config config.json
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import tracemalloc
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================

PIPELINE_VERSION = "1.0.0"
DEFAULT_COMPRESSION = "ZSTD"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# =============================================================================
# EXCEPTIONS
# =============================================================================

class NormalizationError(Exception):
    """Base exception for the normalization pipeline."""
    pass


class ConfigurationError(NormalizationError):
    """Raised when the configuration is invalid or missing."""
    pass


class DatasetError(NormalizationError):
    """Raised when the dataset structure or access fails."""
    pass


class ValidationError(NormalizationError):
    """Raised when the dataset schema fails validation."""
    pass

# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class TransformationConfig(BaseModel):
    """Configuration model for the transformation rules."""
    batch_size: int = Field(50000, gt=0, description="Number of rows per processing batch.")
    compression: str = Field(DEFAULT_COMPRESSION, description="Parquet compression algorithm.")
    minimum_length: int = Field(3, gt=0, description="Minimum valid character length of a question.")
    maximum_length: int = Field(512, gt=0, description="Maximum valid character length of a question.")
    prefixes: List[str] = Field(default_factory=list, description="Conversational prefixes to remove.")
    canonical_replacements: Dict[str, str] = Field(default_factory=dict, description="Phrases to standardize.")
    cleaning_version: str = Field("1.0.0", description="Version identifier for the cleaning logic.")


class Checkpoint(BaseModel):
    """Model representing the state of a processing checkpoint."""
    file_hash: str
    config_hash: str
    last_processed_batch: int
    timestamp: str

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ProcessingStats:
    """Tracks metrics and statistics during pipeline execution."""
    rows_processed: int = 0
    rows_valid: int = 0
    rows_invalid: int = 0
    rows_empty: int = 0
    rows_prefix_removed: int = 0
    rows_canonical_replaced: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def execution_time(self) -> float:
        """Total execution time in seconds."""
        return (self.end_time or time.perf_counter()) - self.start_time

    @property
    def rows_per_second(self) -> float:
        """Throughput in rows per second."""
        exec_time = self.execution_time
        return self.rows_processed / exec_time if exec_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics to a dictionary."""
        return {
            "rows_processed": self.rows_processed,
            "rows_valid": self.rows_valid,
            "rows_invalid": self.rows_invalid,
            "rows_empty": self.rows_empty,
            "rows_prefix_removed": self.rows_prefix_removed,
            "rows_canonical_replaced": self.rows_canonical_replaced,
            "average_processing_time_per_10k": (self.execution_time / max(1, self.rows_processed)) * 10000,
            "rows_per_second": round(self.rows_per_second, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "execution_time_seconds": round(self.execution_time, 2)
        }

# =============================================================================
# UTILITIES
# =============================================================================

def compute_hash(filepath: Path) -> str:
    """Computes a SHA-256 hash of a file for deterministic tracking."""
    if not filepath.exists():
        raise DatasetError(f"File not found for hashing: {filepath}")
    
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_hash(data: Dict[str, Any]) -> str:
    """Computes a SHA-256 hash of a dictionary (used for configuration)."""
    serialized = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()

# =============================================================================
# TRANSFORMATION BASE CLASS
# =============================================================================

class BaseTransformation(ABC):
    """
    Abstract base class for all transformations.
    Transformations must be independently testable and composable.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the transformation."""
        pass

    @abstractmethod
    def apply(self, expr: pl.Expr) -> pl.Expr:
        """
        Applies a transformation using Polars expressions.
        
        Args:
            expr: The Polars expression representing the column to transform.
            
        Returns:
            A new Polars expression with the transformation applied.
        """
        pass

# =============================================================================
# TRANSFORMATION CLASSES
# =============================================================================

class UnicodeNormalization(BaseTransformation):
    """Normalizes unicode characters to NFKC form."""
    name = "unicode_normalization"

    def apply(self, expr: pl.Expr) -> pl.Expr:
        return expr.map_elements(
            lambda x: unicodedata.normalize("NFKC", x) if x is not None else x,
            return_dtype=pl.String
        )


class LowercaseTransformation(BaseTransformation):
    """Converts the text to lowercase."""
    name = "lowercase_transformation"

    def apply(self, expr: pl.Expr) -> pl.Expr:
        return expr.str.to_lowercase()


class PunctuationCleaning(BaseTransformation):
    """
    Removes conversational punctuation but preserves semantic symbols.
    Removes ???, !!!, ..., and multiple commas/periods.
    Preserves alphanumeric characters, spaces, and /, %, -.
    """
    name = "punctuation_cleaning"

    def apply(self, expr: pl.Expr) -> pl.Expr:
        # Keep word chars, whitespace, and specific semantic symbols (/, %, -).
        # Replace everything else with a space to avoid joining words.
        return expr.str.replace_all(r'[^\w\s/%-]', ' ')


class ConversationPrefixRemoval(BaseTransformation):
    """Removes conversational fluff prefixes from the beginning of strings."""
    name = "conversation_prefix_removal"

    def __init__(self, prefixes: List[str]):
        # Sort by length descending so longer prefixes match first
        sorted_prefixes = sorted(prefixes, key=len, reverse=True)
        escaped_prefixes = [re.escape(p.strip().lower()) for p in sorted_prefixes]
        # Regex matches any of the prefixes at the start of the string, optionally followed by whitespace
        self.pattern = r'^(?:' + '|'.join(escaped_prefixes) + r')\s*'

    def apply(self, expr: pl.Expr) -> pl.Expr:
        if not self.pattern or self.pattern == r'^(?:)\s*':
            return expr
        return expr.str.replace(self.pattern, "")


class CanonicalReplacement(BaseTransformation):
    """Replaces non-standard phrases with their canonical equivalents."""
    name = "canonical_replacement"

    def __init__(self, replacements: Dict[str, str]):
        # Sort keys by length descending to prevent partial match overwrites
        self.replacements = dict(sorted(
            replacements.items(), 
            key=lambda item: len(item[0]), 
            reverse=True
        ))

    def apply(self, expr: pl.Expr) -> pl.Expr:
        current_expr = expr
        for old, new in self.replacements.items():
            # Use word boundaries (\b) to avoid replacing sub-words
            pattern = r'\b' + re.escape(old.lower()) + r'\b'
            current_expr = current_expr.str.replace_all(pattern, new.lower())
        return current_expr


class WhitespaceNormalization(BaseTransformation):
    """Collapses multiple spaces, removes tabs/newlines, and trims."""
    name = "whitespace_normalization"

    def apply(self, expr: pl.Expr) -> pl.Expr:
        return (
            expr
            .str.replace_all(r'[\t\n\r]+', ' ')  # Replace tabs/newlines with space
            .str.replace_all(r'\s+', ' ')        # Collapse multiple spaces
            .str.strip_chars()                   # Trim leading/trailing spaces
        )

# =============================================================================
# PIPELINE
# =============================================================================

class NormalizationPipeline:
    """
    Orchestrates the sequential application of transformations on a Polars DataFrame.
    Adheres to Dependency Injection by accepting transformations at initialization.
    """
    
    def __init__(self, transformations: List[BaseTransformation]):
        self.transformations = transformations
        self.logger = logging.getLogger(self.__class__.__name__)

    def process(self, df: pl.DataFrame, source_col: str, target_col: str) -> pl.DataFrame:
        """
        Processes a DataFrame, applying transformations and tracking statistics.
        
        Args:
            df: Input Polars DataFrame.
            source_col: Name of the original question column.
            target_col: Name of the column to store normalized results.
            
        Returns:
            Transformed Polars DataFrame with tracking flags.
        """
        # Initialize target column, safely handling nulls
        df = df.with_columns(pl.col(source_col).fill_null("").alias(target_col))

        for transform in self.transformations:
            before_col = f"{target_col}_before_{transform.name}"
            df = df.with_columns(pl.col(target_col).alias(before_col))
            
            # Apply expression
            df = df.with_columns(transform.apply(pl.col(target_col)).alias(target_col))
            
            # Track changes for statistics
            changed_col = f"changed_{transform.name}"
            df = df.with_columns(
                (pl.col(target_col) != pl.col(before_col)).alias(changed_col)
            )
            
            # Drop intermediate state to preserve memory
            df = df.drop(before_col)

        return df

# =============================================================================
# DATASET PROCESSOR
# =============================================================================

class DatasetNormalizer:
    """
    Primary engine for normalizing the agricultural question dataset.
    Handles schema validation, chunking, checkpointing, and writing to Parquet.
    """

    # Expected input CSV schema mapping (Input Column -> Standardized Output Column)
    SCHEMA_MAPPING = {
        "Id": "id",
        "Question": "original_question",
        "Crop": "crop",
        "Category": "category",
        "Sector": "sector",
        "Location": "location",
        "Date & Time": "date"
    }

    def __init__(self, input_path: Path, output_dir: Path, config_path: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.config_path = config_path
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stats = ProcessingStats()
        
        self._setup_environment()
        self.config = self._load_configuration()
        self.pipeline = self._build_pipeline()

    def _setup_environment(self) -> None:
        """Creates required directories and configures logging."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.output_dir / "normalization.log"
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger.info("Initializing DatasetNormalizer Engine...")

    def _load_configuration(self) -> TransformationConfig:
        """Loads and validates configuration JSON."""
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            config = TransformationConfig(**raw_config)
            self.logger.info(f"Configuration loaded successfully. Batch size: {config.batch_size}")
            return config
        except PydanticValidationError as e:
            raise ConfigurationError(f"Invalid configuration format: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Failed to parse JSON config: {e}")

    def _build_pipeline(self) -> NormalizationPipeline:
        """Constructs the sequential transformation pipeline."""
        transformations = [
            UnicodeNormalization(),
            LowercaseTransformation(),
            WhitespaceNormalization(),
            ConversationPrefixRemoval(self.config.prefixes),
            PunctuationCleaning(),
            WhitespaceNormalization(), # Re-apply to clean up removed punctuation spaces
            CanonicalReplacement(self.config.canonical_replacements),
            WhitespaceNormalization()  # Final trim
        ]
        return NormalizationPipeline(transformations)

    def _get_target_schema(self) -> pa.Schema:
        """Returns the strict PyArrow schema for the canonical Parquet output."""
        return pa.schema([
            ('id', pa.string()),
            ('original_question', pa.string()),
            ('normalized_question', pa.string()),
            ('crop', pa.string()),
            ('category', pa.string()),
            ('sector', pa.string()),
            ('location', pa.string()),
            ('date', pa.string()),
            ('language', pa.string()),
            ('cleaning_version', pa.string()),
            ('processing_timestamp', pa.string()),
            ('is_valid', pa.bool_()),
            ('validation_errors', pa.list_(pa.string()))
        ])

    def _validate_and_map_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validates incoming CSV schema and strips whitespace from headers."""
        # Clean column headers
        df = df.rename({col: col.strip() for col in df.columns})
        
        missing_cols = [col for col in self.SCHEMA_MAPPING.keys() if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing expected columns in chunk: {missing_cols}. Filling with nulls.")
            for col in missing_cols:
                df = df.with_columns(pl.lit(None).alias(col))
        
        # Cast to String and alias to standardized names
        mapping_exprs = [
            pl.col(orig).cast(pl.Utf8).alias(standard) 
            for orig, standard in self.SCHEMA_MAPPING.items()
        ]
        
        return df.select(mapping_exprs)

    def _apply_validations(self, df: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """Applies configurable length, structure, and noise validations."""
        cfg = self.config
        
        # Collect validation errors into a single List[str] column using lazy expressions
        errors = pl.concat_list([
            pl.when(pl.col(col_name) == "").then(pl.lit("empty_question")).otherwise(pl.lit(None)),
            pl.when(pl.col(col_name).str.contains(r'^[\W_]+$')).then(pl.lit("only_punctuation")).otherwise(pl.lit(None)),
            pl.when(pl.col(col_name).str.contains(r'^[0-9\s]+$')).then(pl.lit("only_numbers")).otherwise(pl.lit(None)),
            pl.when(pl.col(col_name).str.len_chars() < cfg.minimum_length).then(pl.lit("too_short")).otherwise(pl.lit(None)),
            pl.when(pl.col(col_name).str.len_chars() > cfg.maximum_length).then(pl.lit("too_long")).otherwise(pl.lit(None))
        ]).list.drop_nulls()

        df = df.with_columns(errors.alias("validation_errors"))
        df = df.with_columns(
            (pl.col("validation_errors").list.len() == 0).alias("is_valid")
        )
        return df

    def _process_chunk(self, df: pl.DataFrame) -> pl.DataFrame:
        """Executes the complete transformation and validation logic on a chunk."""
        df = self._validate_and_map_columns(df)
        
        # Standardize metadata
        df = df.with_columns([
            pl.lit("en").alias("language"),
            pl.lit(self.config.cleaning_version).alias("cleaning_version"),
            pl.lit(datetime.now(timezone.utc).isoformat()).alias("processing_timestamp"),
        ])
        
        # Run Normalization Pipeline
        df = self.pipeline.process(df, source_col="original_question", target_col="normalized_question")
        
        # Update Tracking Statistics
        changed_prefix_col = f"changed_{ConversationPrefixRemoval.name}"
        changed_canon_col = f"changed_{CanonicalReplacement.name}"
        
        if changed_prefix_col in df.columns:
            self.stats.rows_prefix_removed += df[changed_prefix_col].sum()
        if changed_canon_col in df.columns:
            self.stats.rows_canonical_replaced += df[changed_canon_col].sum()

        # Run Validations
        df = self._apply_validations(df, col_name="normalized_question")
        
        # Update Core Stats
        self.stats.rows_processed += len(df)
        valid_count = df["is_valid"].sum()
        self.stats.rows_valid += valid_count
        self.stats.rows_invalid += (len(df) - valid_count)
        
        # Count empty rows specifically
        # A list containing "empty_question" means it was flagged
        empty_expr = pl.col("validation_errors").list.contains("empty_question")
        self.stats.rows_empty += df.select(empty_expr).sum().item()

        # Select final ordered columns matching PyArrow schema
        final_columns = [field.name for field in self._get_target_schema()]
        return df.select(final_columns)

    def _handle_checkpoint(self, file_hash: str, config_hash: str) -> int:
        """Reads checkpoint file to determine if processing can be resumed."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return 0
            
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ckpt = Checkpoint(**data)
                
            if ckpt.file_hash == file_hash and ckpt.config_hash == config_hash:
                self.logger.info(f"Resuming from batch {ckpt.last_processed_batch} via checkpoint.")
                return ckpt.last_processed_batch
            else:
                self.logger.info("File or configuration changed. Ignoring previous checkpoint.")
                return 0
        except Exception as e:
            self.logger.warning(f"Failed to read checkpoint, starting fresh. Reason: {e}")
            return 0

    def _save_checkpoint(self, file_hash: str, config_hash: str, current_batch: int) -> None:
        """Persists progress to disk."""
        ckpt = Checkpoint(
            file_hash=file_hash,
            config_hash=config_hash,
            last_processed_batch=current_batch,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(ckpt.model_dump(), f, indent=4)

    def _write_metadata_and_stats(self) -> None:
        """Writes final processing statistics and dataset metadata to disk."""
        self.stats.end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        self.stats.peak_memory_mb = peak_mem / (1024 * 1024)
        
        # Write Processing Statistics
        stats_path = self.output_dir / "processing_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats.to_dict(), f, indent=4)

        # Write Metadata
        metadata = {
            "pipeline_version": PIPELINE_VERSION,
            "cleaning_version": self.config.cleaning_version,
            "processing_date": datetime.now(timezone.utc).isoformat(),
            "rows_processed": self.stats.rows_processed,
            "rows_valid": self.stats.rows_valid,
            "rows_invalid": self.stats.rows_invalid,
            "input_file": str(self.input_path.resolve()),
            "output_file": str((self.output_dir / "questions_cleaned.parquet").resolve()),
            "compression": self.config.compression,
            "language": "en"
        }
        meta_path = self.output_dir / "questions_cleaned.metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        self.logger.info(f"Saved metadata and statistics to {self.output_dir}")

    def run(self) -> None:
        """Main execution entry point."""
        self.logger.info(f"Starting Normalization Pipeline for {self.input_path}")
        tracemalloc.start()
        
        file_hash = compute_hash(self.input_path)
        config_hash = compute_dict_hash(self.config.model_dump())
        
        start_batch = self._handle_checkpoint(file_hash, config_hash)
        output_parquet = self.output_dir / "questions_cleaned.parquet"
        
        # If starting fresh, remove old output file if it exists
        if start_batch == 0 and output_parquet.exists():
            output_parquet.unlink()

        # Initialize PyArrow ParquetWriter
        schema = self._get_target_schema()
        writer = None

        try:
            # Polars Batched Reader for Memory-Efficient Streaming
            reader = pl.read_csv_batched(
                str(self.input_path),
                batch_size=self.config.batch_size,
                infer_schema_length=10000,
                ignore_errors=True
            )
            
            current_batch = 0
            
            while True:
                batches = reader.next_batches(1)
                if not batches:
                    break
                
                chunk_df = batches[0]
                current_batch += 1
                
                if current_batch <= start_batch:
                    # Skip previously processed batches
                    continue
                    
                processed_chunk = self._process_chunk(chunk_df)
                
                # Convert to Arrow Table and Append to Parquet
                arrow_table = processed_chunk.to_arrow()
                # Ensure arrow_table matches schema exactly
                arrow_table = arrow_table.cast(schema)
                
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_parquet, 
                        schema, 
                        compression=self.config.compression.upper()
                    )
                
                writer.write_table(arrow_table)
                self._save_checkpoint(file_hash, config_hash, current_batch)
                
                self.logger.info(f"Successfully processed batch {current_batch} ({len(processed_chunk)} rows)")
                
        except Exception as e:
            self.logger.error(f"Pipeline failed at batch {current_batch + 1}: {str(e)}", exc_info=True)
            raise DatasetError(f"Processing failed: {e}")
        finally:
            if writer:
                writer.close()
            tracemalloc.stop()
            self._write_metadata_and_stats()

        self.logger.info("Pipeline Execution Completed Successfully.")
        self.logger.info(f"Metrics: {json.dumps(self.stats.to_dict(), indent=2)}")

# =============================================================================
# CLI / MAIN
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Normalizer Engine for Semantic Topic Discovery."
    )
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True, 
        help="Path to the raw input CSV dataset."
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True, 
        help="Directory to save the cleaned parquet and metadata."
    )
    parser.add_argument(
        "--config", 
        type=Path, 
        required=True, 
        help="Path to the JSON configuration file."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        engine = DatasetNormalizer(
            input_path=args.input,
            output_dir=args.output,
            config_path=args.config
        )
        engine.run()
    except Exception as e:
        logging.getLogger("main").fatal(f"Fatal execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
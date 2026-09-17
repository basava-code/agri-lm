"""
Production-Grade AI Normalization Config Pipeline

Processes large batch CSV query datasets by passing full, untruncated sample questions
directly to a local Ollama instance (Gemma) to discover generic prefixes and canonical term replacements.

Usage:
    python pipeline.py --data-dir data --config config.json --output new_config.json --model gemma3:27b
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

# Configure Timestamped Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DataPipeline")


# 1. PYDANTIC SCHEMAS
class LLMExtractionResponse(BaseModel):
    """Schema expected from Ollama structured JSON extraction output."""
    prefixes: List[str] = Field(
        default_factory=list,
        description="Universal, generic query starting phrases to strip (e.g., 'kripya mujhe batayein', 'farmer asked about')",
    )
    canonical_replacements: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping dictionary for vernacular terms, misspelled crops, pests, diseases, or scheme names.",
    )


class ConfigSchema(BaseModel):
    """Schema for global normalization configuration file."""
    prefixes: List[str] = Field(default_factory=list)
    canonical_replacements: Dict[str, str] = Field(default_factory=dict)
    minimum_length: int = Field(default=3)
    maximum_length: int = Field(default=1000)
    batch_size: int = Field(default=1000)

    class Config:
        extra = "allow"


def to_dict(model: BaseModel) -> Dict[str, Any]:
    """Helper for cross-version Pydantic compatibility (v1 & v2)."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()



# 2. PRE-PROCESSOR (FULL QUESTION SAMPLING)
class PreProcessor:
    """Handles CSV loading and sampling full, unchopped original queries for LLM analysis."""

    def __init__(self, sample_size: int = 150):
        self.sample_size = sample_size

    def extract_queries_from_csv(self, file_path: Path) -> List[str]:
        """Reads CSV and extracts raw question strings from target columns."""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if df.empty:
                logger.warning(f"File {file_path.name} is empty.")
                return []

            candidate_cols = ["Question", "question", "Query", "query", "text", "Text", "Prompt", "prompt"]
            target_col = None
            for col in candidate_cols:
                if col in df.columns:
                    target_col = col
                    break

            if target_col is None:
                target_col = df.columns[0]
                logger.info(f"Target column not explicitly matched in {file_path.name}. Defaulting to first column: '{target_col}'")

            queries = df[target_col].dropna().astype(str).str.strip().tolist()
            # Clean excessive whitespace while keeping original sentence context
            cleaned_queries = [re.sub(r"\s+", " ", q) for q in queries if len(q.strip()) > 3]
            return cleaned_queries

        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return []

    def sample_full_queries(self, queries: List[str]) -> List[str]:
        """
        Deduplicates and samples complete original queries directly without text reduction.
        """
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        sample_count = min(self.sample_size, len(unique_queries))
        if len(unique_queries) <= sample_count:
            return unique_queries

        # Evenly spaced deterministic sampling across the batch
        step = max(1, len(unique_queries) // sample_count)
        return unique_queries[::step][:sample_count]


# 3. OLLAMA LLM CLIENT
class OllamaClient:
    """Handles communication with local Ollama LLM server with backoff retries and JSON format enforcement."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:27b",
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def generate_normalization_rules(
        self, full_queries: List[str]
    ) -> Optional[LLMExtractionResponse]:
        """Sends full question sample directly to Ollama Gemma and returns parsed normalization rules."""
        endpoint = f"{self.base_url}/api/generate"
        prompt_content = self._build_prompt(full_queries)

        payload = {
            "model": self.model,
            "prompt": prompt_content,
            "stream": False,
            "format": "json",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Calling Ollama API (Model: '{self.model}', Attempt {attempt}/{self.max_retries})..."
                )
                response = requests.post(endpoint, json=payload, timeout=120)
                response.raise_for_status()

                res_json = response.json()
                raw_response_text = res_json.get("response", "")

                cleaned_json_text = self._clean_json_string(raw_response_text)
                parsed_data = json.loads(cleaned_json_text)

                validated_response = LLMExtractionResponse(**parsed_data)
                return validated_response

            except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Ollama call attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor**attempt
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Exhausted all retries for Ollama API call.")
                    return None

    def _clean_json_string(self, text: str) -> str:
        """Strips markdown code block backticks if present in LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _build_prompt(self, full_queries: List[str]) -> str:
        """Constructs prompt using full, untruncated original questions."""
        formatted_queries = "\n".join([f"- {q}" for q in full_queries])

        return f"""
You are an expert NLP Data Normalization Engineer specializing in multilingual agricultural datasets.

You are analyzing ORIGINAL, UNTRUNCATED user queries.

------------------------------------------------------------
ORIGINAL QUERIES
------------------------------------------------------------

{formatted_queries}

------------------------------------------------------------
OBJECTIVE
------------------------------------------------------------

Your goal is to discover reusable normalization rules.

Return ONLY two things:

1. Generic conversational prefixes
2. Canonical replacement mappings

These rules will later be applied to MILLIONS of unseen agricultural queries.

Therefore they must GENERALIZE well.

Never overfit to this batch.

------------------------------------------------------------
PART 1 — PREFIX EXTRACTION
------------------------------------------------------------

A prefix is ONLY conversational padding that appears BEFORE the real question.

A prefix contributes ZERO domain meaning.

Removing the prefix MUST NOT change the meaning of the actual agricultural question.

Example:

Input:
Farmer wants to know information about wheat rust disease.

Correct prefix:
Farmer wants to know information about

Remaining query:
wheat rust disease

This is GOOD because the remaining query still preserves the complete intent.

------------------------------------------------------------
WHAT IS NOT A PREFIX
------------------------------------------------------------

Never include words that belong to the actual agricultural intent.

These include (not exhaustive):

• crop names
• pest names
• disease names
• fertilizer names
• weather
• rainfall
• irrigation
• subsidy
• PM-Kisan
• market price
• sowing
• harvesting
• spraying
• seed treatment
• nutrient management
• varieties
• insect control
• weed control
• cultivation
• livestock
• vaccination
• milk production
• horticulture
• government schemes

------------------------------------------------------------
BAD PREFIX EXAMPLES
------------------------------------------------------------

❌ how to control

Reason:
"control" is the intent.

------------------------------------------------------------

❌ query about sowing time of

Reason:
"sowing time" is semantic content.

------------------------------------------------------------

❌ asked about weather report of

Reason:
weather report is the user's intent.

------------------------------------------------------------

❌ market price of

Reason:
market price is the topic.

------------------------------------------------------------

❌ information regarding weather

Reason:
weather must remain.

------------------------------------------------------------

❌ which fertilizer to use in

Reason:
fertilizer recommendation is the query.

------------------------------------------------------------

❌ control measure leaf curling of brinjal

Reason:
Entire agricultural meaning has been consumed.

------------------------------------------------------------

❌ I want to grow

Reason:
"grow" is the intent.

------------------------------------------------------------
GOOD PREFIX EXAMPLES
------------------------------------------------------------

✓ farmer asked

✓ farmer asked about

✓ farmer wants to know

✓ farmer wants information about

✓ farmer wants detailed information about

✓ please tell me

✓ kindly tell me

✓ kripya mujhe batayein

✓ sir mujhe janna hai

✓ give me information

✓ information regarding

✓ query regarding

✓ query about

✓ asked about

✓ want to know

✓ can you tell me

✓ please guide me

These contain NO agricultural topic.

------------------------------------------------------------
PREFIX VALIDATION RULE
------------------------------------------------------------

Before adding ANY prefix ask:

If I remove this phrase,
does the remaining sentence still contain the COMPLETE agricultural topic?

If NO

DO NOT include it.

------------------------------------------------------------
PART 2 — CANONICAL REPLACEMENTS
------------------------------------------------------------

Extract reusable mappings for:

• spelling mistakes
• OCR mistakes
• transliteration
• Hindi words
• regional crop names
• local pest names
• fertilizer abbreviations
• government schemes
• agricultural terminology
• common abbreviations

Good examples:

"mandi bhav" -> "market price"

"aalu" -> "potato"

"dhaan" -> "paddy"

"makka" -> "maize"

"forecost" -> "forecast"

"pm kisan samman nidhi" -> "pm-kisan"

------------------------------------------------------------
CANONICAL MAPPING RULES
------------------------------------------------------------

Mappings must:

✓ be reusable

✓ preserve meaning

✓ never invent facts

✓ never summarize

✓ never merge unrelated concepts

✓ never translate incorrectly

✓ be useful on unseen data

------------------------------------------------------------
DEDUPLICATION
------------------------------------------------------------

For prefixes:

• lowercase
• trim spaces
• remove duplicates
• remove punctuation variants
• keep the shortest reusable form

Example

Bad:
farmer wants information about the
farmer wants information about

Output only:

farmer wants information about

------------------------------------------------------------
QUALITY FILTER
------------------------------------------------------------

DO NOT output any prefix if it:

contains a crop

contains a pest

contains a disease

contains fertilizer

contains weather

contains market

contains irrigation

contains sowing

contains harvesting

contains spraying

contains seed

contains scheme names

contains cultivation

contains livestock

contains any agricultural noun

contains the user's intent

When uncertain,

DO NOT include it.

------------------------------------------------------------
CRITICAL OUTPUT REQUIREMENT:
Respond STRICTLY with a valid JSON object matching this schema:
{{
  "prefixes": ["phrase 1", "phrase 2"],
  "canonical_replacements": {{
    "non_standard_term": "canonical_term"
  }}
}}
"""


# 4. CONFIG MANAGER & ATOMIC PERSISTENCE
class ConfigManager:
    """Manages merging, rule sanitization, prefix length sorting, and atomic updates."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> ConfigSchema:
        """Loads existing configuration or creates a default instance."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded existing configuration from {self.config_path}")
                return ConfigSchema(**data)
            except Exception as e:
                logger.warning(
                    f"Failed to load existing config from {self.config_path}: {e}. Creating fallback config."
                )
        return ConfigSchema()

    def _is_valid_prefix(self, prefix: str) -> bool:
        """Filters out corrupted rules or LLM artifacts from prefixes."""
        p = prefix.strip().lower()
        if not p or len(p) < 2 or len(p) > 80:
            return False
        # Reject markdown comments or LLM meta-talk
        invalid_patterns = [r"//", r"/\*", r"\{", r"\}", r"placeholder", r"single-word", r"1/10", r"let me refine"]
        for pattern in invalid_patterns:
            if re.search(pattern, p):
                return False
        return True

    def _is_valid_replacement(self, key: str, val: str) -> bool:
        """Filters out corrupted rules or LLM artifacts from replacements."""
        k, v = key.strip().lower(), val.strip().lower()
        if not k or not v or k == v:
            return False
        if len(k) > 100 or len(v) > 100:
            return False
        invalid_patterns = [r"//", r"/\*", r"\{", r"\}", r"placeholder", r"single-word", r"mistake", r"refine"]
        for pattern in invalid_patterns:
            if re.search(pattern, k) or re.search(pattern, v):
                return False
        return True

    def merge_and_save(self, extraction: LLMExtractionResponse, output_path: Path) -> None:
        """
        Merges new rules into configuration, strictly sorts prefixes in descending order of length,
        and atomically writes output to prevent write corruption.
        """
        # 1. Merge and Deduplicate Prefixes
        existing_prefixes = set(self.config.prefixes)
        for prefix in extraction.prefixes:
            cleaned_p = prefix.strip().lower()
            if self._is_valid_prefix(cleaned_p):
                existing_prefixes.add(cleaned_p)

        # CRITICAL RULE: Sort prefixes in DESCENDING ORDER OF LENGTH (longest phrase first)
        sorted_prefixes = sorted(list(existing_prefixes), key=len, reverse=True)

        # 2. Merge Canonical Replacements Dictionary
        existing_replacements = dict(self.config.canonical_replacements)
        for key, val in extraction.canonical_replacements.items():
            k_clean = key.strip().lower()
            v_clean = val.strip().lower()
            if self._is_valid_replacement(k_clean, v_clean):
                existing_replacements[k_clean] = v_clean

        # Update Config State
        self.config.prefixes = sorted_prefixes
        self.config.canonical_replacements = existing_replacements

        # Validate with Pydantic Schema
        validated_config = ConfigSchema(**to_dict(self.config))

        # 3. Atomic Write (Write to .tmp then replace)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

        try:
            with open(tmp_output_path, "w", encoding="utf-8") as f:
                json.dump(to_dict(validated_config), f, indent=2, ensure_ascii=False)

            tmp_output_path.replace(output_path)
            logger.info(
                f"Successfully updated configuration atomically saved to '{output_path}' "
                f"(Total Prefixes: {len(validated_config.prefixes)}, "
                f"Total Replacements: {len(validated_config.canonical_replacements)})"
            )
        except Exception as e:
            logger.error(f"Failed to atomically write configuration to {output_path}: {e}")
            if tmp_output_path.exists():
                tmp_output_path.unlink()


# 5. BATCH PIPELINE ORCHESTRATOR
class BatchPipeline:
    """Orchestrates CSV iteration, full query sampling, LLM inference, and atomic updates."""

    def __init__(
        self,
        data_dir: Path,
        config_path: Path,
        output_path: Path,
        model: str,
        ollama_url: str,
        sample_size: int,
    ):
        self.data_dir = data_dir
        self.config_path = config_path
        self.output_path = output_path

        self.preprocessor = PreProcessor(sample_size=sample_size)
        self.ollama_client = OllamaClient(base_url=ollama_url, model=model)
        self.config_manager = ConfigManager(config_path=config_path)

    def run(self) -> None:
        """Executes pipeline over all CSV files in target directory."""
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            logger.error(f"Input directory does not exist or is not a directory: {self.data_dir}")
            return

        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        if not csv_files:
            logger.warning(f"No CSV files found in directory: {self.data_dir}")
            return

        logger.info(f"Discovered {len(csv_files)} CSV batch file(s) in {self.data_dir}")

        for csv_file in tqdm(csv_files, desc="Processing Batches", unit="file"):
            logger.info(f"=== Processing Batch File: {csv_file.name} ===")

            queries = self.preprocessor.extract_queries_from_csv(csv_file)
            if not queries:
                logger.warning(f"Skipping empty or unparseable batch: {csv_file.name}")
                continue

            full_sample_queries = self.preprocessor.sample_full_queries(queries)
            logger.info(f"Loaded {len(queries)} queries. Extracted {len(full_sample_queries)} full original query samples for LLM.")

            extraction = self.ollama_client.generate_normalization_rules(
                full_queries=full_sample_queries
            )

            if extraction:
                logger.info(
                    f"LLM Discovered Rules: {len(extraction.prefixes)} prefix(es), "
                    f"{len(extraction.canonical_replacements)} replacement rule(s)."
                )
                self.config_manager.merge_and_save(extraction, self.output_path)
            else:
                logger.error(f"Extraction failed for batch {csv_file.name}. Continuing to next file.")

        logger.info("Batch processing pipeline executed successfully.")


# 6. CLI ENTRY POINT
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Normalization Pipeline using Full Question Sampling and Ollama LLM."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to input directory containing CSV files (default: data)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to initial master configuration file (default: config.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("new_config.json"),
        help="Path to output updated configuration file (default: new_config.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:27b",
        help="Ollama model tag to run (default: gemma3:27b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Local Ollama instance URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150,
        help="Number of full original query samples sent to Ollama per batch (default: 150)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Initializing Batch Processing Pipeline...")
    logger.info(f"Data Dir   : {args.data_dir.resolve()}")
    logger.info(f"Config File: {args.config.resolve()}")
    logger.info(f"Output File: {args.output.resolve()}")
    logger.info(f"Model      : {args.model}")
    logger.info(f"Ollama URL : {args.ollama_url}")

    pipeline = BatchPipeline(
        data_dir=args.data_dir,
        config_path=args.config,
        output_path=args.output,
        model=args.model,
        ollama_url=args.ollama_url,
        sample_size=args.sample_size,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
# 🌾 Kisan Call Center (KCC) High-Throughput Data Engineering Pipeline

The **KCC Pipeline** is a production data engineering and synthetic dialogue generation pipeline built to ingest, sanitize, normalize, and transform millions of raw Indian farmer call queries from [Data.gov.in](https://data.gov.in) into high-fidelity conversational instruction datasets.

KCC records represent real-world ground truth from across 28 states and Union territories. However, raw telephonic records are fraught with telephonic operator greetings (*"Namaskar Kisan bhai, KCC mein aapka swagat hai"*), speech-to-text transcription artifacts, and vernacular crop names that break conventional LLM tokenizers. This pipeline resolves these challenges at scale.

---

## 🏗️ Pipeline Architecture

```mermaid
flowchart LR
    A["Data.gov.in API\n(Monthly Dumps)"] --> B["1. Downloader\n(Async HTTPX Scraper)"]
    B --> C["2. Telephonic Filter\n(Java 21 Loom Filter)"]
    C --> D["3. Streaming Cleaner\n(50k Chunk CSV Cleaner)"]
    D --> E["4. Taxonomy Normalizer\n(Polars + PyArrow)"]
    E --> F["5. Data Dictionary\n(Gemma-3 27B Ollama)"]
    F --> G["6. Dialogue Generator\n(vLLM FP8 Teacher)"]
    G --> H["Pristine SFT Dataset\n(Multiturn Agronomic Q&A)"]
```

---

## 📂 Subsystem Modules & Directory Guide

| Subsystem / Directory | Technology Stack | Core Purpose | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **[`downloader/`](./downloader/)** | Python `httpx`, `asyncio` | Resumable Data.gov.in API scraper with durable page checkpoints. | API Key & Dataset ID | Raw CSV batch files (100k rows each) |
| **[`java-filter/`](./java-filter/)** | Java 21 Virtual Threads (`Project Loom`) | Sub-millisecond telephonic noise classifier handling up to 5,000 req/s. | Raw CSV batch files | Noise-free agronomic text |
| **[`cleaning/`](./cleaning/)** | Python `pandas`, `unicodedata`, `hashlib` | 50k streaming chunk CSV cleaner with 64-bit query hash deduplication. | Raw CSV batches | Cleaned CSV files (11 canonical columns) |
| **[`normalization/`](./normalization/)** | `polars`, `pyarrow`, `pydantic` | High-speed taxonomic mapper converting colloquial terms to botanical standard. | Cleaned CSVs + `config.json` | Zstandard-compressed Parquet |
| **[`data-dictionary/`](./data-dictionary/)** | Ollama, Gemma-3 27B, `pydantic` | Automated AI discovery of emerging prefixes and vernacular crop names. | Sample CSV batches | Enriched `config.json` rules |
| **[`generation/`](./generation/)** | vLLM, OpenAI SDK, FP8 Gemma-4 | Asynchronous synthesis of multi-turn farmer-expert advisory dialogues. | Cleaned queries + answers | Multi-turn instruction JSONL |

---

## ⚡ Execution Pipeline Progression

To execute the entire pipeline from raw web ingestion to fine-tuning ready dialogues, follow this sequence:

### Step 1: Download Monthly Records from Data.gov.in
```bash
cd downloader
# Set DATA_GOV_API_KEY and DATASET_ID in .env
python cli.py
```

### Step 2: High-Concurrency Telephonic Cleaning (Java 21 Loom)
```bash
cd ../java-filter
mvn clean package -DskipTests
java -jar target/java-filter-1.0-SNAPSHOT.jar
```

### Step 3: Streaming Chunk Deduplication & Cleaning
```bash
cd ../cleaning
python cleaner.py \
    --input-dir ../downloader/data/csv \
    --output-dir ../data/cleaned \
    --chunk-size 50000
```

### Step 4: Discover & Enrich Taxonomic Rules
```bash
cd ../data-dictionary
python pipeline.py \
    --data-dir ../data/cleaned \
    --config ../normalization/config.json \
    --output ../normalization/config.json \
    --model gemma3:27b
```

### Step 5: Semantic Normalization & Botanical Mapping
```bash
cd ../normalization
python dataset_normalizer.py \
    --input ../data/cleaned/cleaned_records.csv \
    --output ../data/normalized \
    --config config.json
```

### Step 6: Multi-Turn Dialogue Synthesis with vLLM
```bash
cd ../generation
python generate_data_vllm.py
```

---

## 🔒 Production Resilience & Fault Tolerance
- **Graceful Interrupt Handling:** All scripts intercept `SIGINT` (`Ctrl+C`) and write atomic checkpoints before exiting, ensuring no progress loss on multi-gigabyte datasets.
- **Zero Memory Leaks:** Streaming chunk processing prevents out-of-memory (OOM) errors even on 50M+ record dumps.
- **Type-Safe Invariants:** Every stage enforces Pydantic data schemas and PyArrow types at runtime.

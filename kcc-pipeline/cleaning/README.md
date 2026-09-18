# 🧹 KCC Streaming Chunk ETL Cleaner & Deduplicator

High-throughput, memory-efficient data cleaner designed to process multi-gigabyte raw KCC CSV files in streaming batches, removing telephonic noise, standardizing encodings, and deduplicating queries.

---

## 🛠️ Key Technical Capabilities

- **Streaming Batch Processing:** Iterates over datasets in configurable chunks (default: 50,000 rows), ensuring peak RAM consumption stays under 500 MB regardless of dataset size.
- **64-bit Query Deduplication:** Computes MD5/SHA256 query hashes to eliminate repeated identical inquiries while tracking duplicate frequency counts.
- **Encoding Normalization:** Gracefully decodes diverse legacy encodings (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`) and normalizes Unicode diacritics via `unicodedata.normalize('NFKD')`.
- **Text Standardizer:** Strips leading telephonic operator filler phrases, normalizes irregular whitespace, and restores proper sentence capitalization.
- **Atomic Resume Checkpoints:** Maintains state in `state/checkpoint.json` with a `SIGINT` (`Ctrl+C`) signal handler, allowing instant resumption after interruptions.

---

## 📊 Schema Transformation

The cleaner validates raw inputs and maps them into an 11-column standardized analytical schema:

| Raw Data.gov.in Column | Cleaned Canonical Column | Description |
| :--- | :--- | :--- |
| *Auto-Generated* | `Id` | Unique record hash identifier |
| `QueryText` | `Question` | Sanitized farmer inquiry |
| `KccAns` | `Answer` | Cleaned agricultural expert advisory |
| `Crop` | `Crop` | Target crop name |
| `Category` | `Category` | High-level agronomic category |
| `Sector` | `Sector` | Agricultural domain sector |
| `QueryType` | `QueryType` | Intent classification (Disease, Nutrient, Weather, etc.) |
| `CreatedOn` / `day`, `month`, `year` | `Date & Time` | Standardized ISO-8601 timestamp |
| `BlockName` | `Block` | Administrative Sub-district / Block |
| `DistrictName` | `District` | District jurisdiction |
| `StateName` | `State` | Indian State or Union Territory |

---

## 🚀 CLI Execution

```bash
# Basic usage
python cleaner.py \
    --input-dir ../downloader/data/csv \
    --output-dir ../../data/cleaned \
    --chunk-size 50000

# Reset state and re-process all raw batches from scratch
python cleaner.py \
    --input-dir ../downloader/data/csv \
    --output-dir ../../data/cleaned \
    --chunk-size 50000 \
    --reset-state \
    -v
```

### CLI Arguments:
- `--input-dir`: Directory containing raw batch CSV files (default: `data/raw`).
- `--output-dir`: Target directory for sanitized CSV files (default: `data/processed`).
- `--chunk-size`: Number of rows per processing slice (default: `50000`).
- `--reset-state`: Deletes existing checkpoint and starts from the first file.
- `-v, --verbose`: Enables verbose debug-level logging.

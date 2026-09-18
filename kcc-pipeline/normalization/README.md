# 🏷️ Semantic Taxonomic Normalization Engine

High-performance dataset normalization engine built on **Polars** and **PyArrow** for the Semantic Topic Discovery Pipeline. Standardizes colloquial regional terminology, removes noise prefixes, and outputs type-safe Parquet files optimized for embedding generation (e.g., `BGE-M3`).

---

## ⚡ Performance & Design Principles

- **Polars Lazy Engine:** Employs zero-copy, columnar lazy evaluation to maximize CPU cache locality and process millions of rows at memory-bus speeds.
- **PyArrow Streaming:** Writes partitioned Parquet files incrementally with Zstandard (`ZSTD`) compression.
- **Pydantic Validation:** Validates pipeline schema integrity and ensures configuration parameters strictly adhere to expected types.
- **Vernacular-to-Botanical Mapping:** Maps regional vernacular crop variants (e.g., *Makki* $\rightarrow$ *Zea mays* / Maize, *Bhutte* $\rightarrow$ Corn, *Sarson* $\rightarrow$ Mustard) into canonical agricultural taxonomies.

---

## ⚙️ Configuration (`config.json`)

The normalizer consumes a declarative JSON configuration containing prefix stripping lists and canonical dictionary mappings:

```json
{
  "prefixes": [
    "kripya mujhe batayein",
    "farmer asked about",
    "kisan ne pucha",
    "information required regarding"
  ],
  "canonical_replacements": {
    "makki": "maize",
    "bhutta": "maize",
    "sarson": "mustard",
    "toria": "rapeseed",
    "dhan": "paddy"
  },
  "minimum_length": 5,
  "maximum_length": 1000,
  "batch_size": 25000
}
```

---

## 🚀 CLI Usage

```bash
# Normalize a cleaned CSV dataset to Parquet
python dataset_normalizer.py \
    --input ../../data/cleaned/cleaned_records.csv \
    --output ../../data/normalized/ \
    --config config.json
```

### CLI Options:
- `--input`: Path to input CSV or folder of CSV files.
- `--output`: Output directory where compressed Parquet partitions will be saved.
- `--config`: Path to `config.json` containing taxonomic mappings and prefixes.
- `--compression`: Parquet compression algorithm (`ZSTD`, `SNAPPY`, `GZIP`; default: `ZSTD`).

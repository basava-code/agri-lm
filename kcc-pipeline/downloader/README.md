# 📥 Data.gov.in Resumable KCC Downloader

Production-grade, asynchronous downloader engineered to reliably scrape multi-gigabyte monthly Kisan Call Center (KCC) datasets from the Indian Government's [Data.gov.in](https://data.gov.in) Open Data API.

---

## ⚡ Architectural Principles & Rate-Limit Resistance

1. **Concurrency Control (`CONCURRENT_REQUESTS = 1`):**
   - The Data.gov.in API load balancer aggressively throttles and returns `HTTP 429 (Too Many Requests)` when concurrent requests are detected. The downloader enforces sequential fetching by default to ensure uninterrupted continuous operation.
2. **Page-by-Page JSON Checkpointing:**
   - Every 1,000-row API page is immediately written to disk as a durable raw CSV part.
   - `checkpoint.json` is updated atomically after every page. If the process is halted or loses network connectivity, it resumes precisely at the last fetched offset.
3. **True CSV Parser Row Counting:**
   - Rather than naive newline counting (which fractures on multiline user query fields), Python's internal CSV parser counts records accurately.
4. **100,000-Row Batch Consolidation:**
   - Pages are merged into unified 100k-row batch files only after all constituent parts are verified durable on disk.

---

## ⚙️ Configuration & Environment

Create a `.env` file in this directory with the following variables:

```env
# Required
DATA_GOV_API_KEY=your_api_key_here
DATASET_ID=your_dataset_resource_id

# Optional (Defaults shown)
PAGE_LIMIT=1000
BATCH_SIZE=100000
CONCURRENT_REQUESTS=1
```

---

## 🚀 Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Start or resume the download
python cli.py
```

### Directory Structure Generated During Execution:
```
downloader/
├── data/
│   ├── parts/                 # Durable page-level CSV chunks (page_0001.csv, ...)
│   └── csv/                   # Consolidated 100,000-row batch files
├── checkpoint.json            # Atomic resume state tracker
└── cli.py                     # Downloader script
```

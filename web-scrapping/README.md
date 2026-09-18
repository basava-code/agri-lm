# 🌐 Web Scraping & Real-Time Policy Harvester Suite

The **Web Scraping Suite** automates continuous data acquisition from Indian agricultural government portals and official press bureaus, feeding real-time policy updates, market advisories, and historical query archives into Agri-LM.

---

## 🏗️ Harvester Modules

```
web-scrapping/
├── README.md               # Master web scraping overview (this file)
├── kcc/                    # High-throughput Data.gov.in KCC queue scraper
│   ├── README.md           # KCC scraper documentation
│   ├── kcc_scrapper.py     # Asynchronous producer-consumer FastAPI scraper
│   └── requirements.txt    # Python dependencies
└── pib/                    # Press Information Bureau (PIB) press release harvester
    ├── README.md           # PIB scraper documentation
    ├── scrapper.py         # ASP.NET ViewState FastAPI scraper
    └── requirements.txt    # Python dependencies
```

---

## 📋 Module Summary

| Harvester | Target Source | Technology | Core Capability | Output Format |
| :--- | :--- | :--- | :--- | :--- |
| **[`kcc/`](./kcc/)** | Data.gov.in API | FastAPI, `httpx`, `asyncio.Queue` | High-performance asynchronous producer-consumer queue with semaphore throttling. | 100,000-row segmented CSV batch files |
| **[`pib/`](./pib/)** | PIB (pib.gov.in) | FastAPI, `httpx`, `BeautifulSoup4` | Navigates ASP.NET forms, extracts dynamic `__VIEWSTATE` tokens, scrapes date ranges. | Monthly consolidated CSV press releases |

---

## 🚀 Quickstart

### 1. KCC Harvester Service
```bash
cd kcc
pip install -r requirements.txt
# Configure DATA_GOV_API_KEY and DATASET_ID in .env
python kcc_scrapper.py
```

### 2. PIB Harvester Service
```bash
cd pib
pip install -r requirements.txt
python scrapper.py
```
Access the interactive FastAPI Swagger UI documentation at `http://localhost:8000/docs`.

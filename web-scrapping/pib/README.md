# 📰 Press Information Bureau (PIB) Press Release Harvester

FastAPI-driven web scraping service engineered to harvest official agricultural press releases, Cabinet Committee on Economic Affairs (CCEA) decisions, Minimum Support Price (MSP) notifications, and fertilizer subsidy announcements directly from the [Press Information Bureau of India (PIB)](https://pib.gov.in).

---

## ⚡ Technical Challenges & Engineering Solutions

The PIB web portal is built on legacy Microsoft ASP.NET Web Forms, which introduces unique challenges:
1. **Dynamic Hidden ViewState:** Form submissions require extracting and reposting large encrypted hidden fields (`__VIEWSTATE`, `__EVENTVALIDATION`, `__VIEWSTATEGENERATOR`).
2. **PostBack Navigation:** Changing the target Ministry or date triggers an asynchronous `__doPostBack` event rather than a clean RESTful URL.

### Our Solution:
- `scrapper.py` uses `BeautifulSoup` to parse and extract the dynamic ViewState payload for every request cycle.
- Maintains persistent HTTP session cookies via `httpx.AsyncClient`.
- Traverses paginated date ranges asynchronously while bundling output records into monthly CSV datasets.

---

## 🚀 Usage & API Endpoints

### 1. Launch the Service:
```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python scrapper.py
```
The server will run on `http://localhost:8000`.

### 2. Available Endpoints:

#### Single-Day Scrape:
```http
GET /scrape?ministry=Ministry of Agriculture & Farmers Welfare&day=15&month=08&year=2025
```

#### Date-Range Batch Scrape:
```http
GET /scrape-range?ministry=Ministry of Agriculture & Farmers Welfare&start_date=01-01-2025&end_date=31-12-2025
```

Outputs are automatically saved into the `data/` directory as CSV files containing title, publication timestamp, target ministry, release ID, and full article text.

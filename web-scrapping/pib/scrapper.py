"""
PIB Press Release Scraper API

This module provides a FastAPI-based web scraping application designed to extract 
press releases from the Press Information Bureau (PIB) of India. It navigates 
the site's ASP.NET forms, handles hidden ViewState tokens, and fetches detailed 
article descriptions concurrently.

Endpoints:
GET /scrape
      Scrapes data for a single specific date and ministry.
      Example: /scrape?ministry=Prime Minister's Office&day=24&month=05&year=2026

GET /scrape-range
      Scrapes data over a specified date range, bundling output files by month.
      Example: /scrape-range?ministry=Ministry of Finance&start_date=01-01-2026&end_date=15-02-2026

Usage:
    python scrapper.py
"""

import os
import csv
import logging
import asyncio
import re
import calendar
from datetime import datetime, timedelta
from urllib.parse import urljoin
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PIB Scraper Testing App",
    description="A single-file FastAPI app to test scraping PIB press releases.",
    version="1.2.0"
)

BASE_URL = "https://www.pib.gov.in/allRel.aspx?reg=46&lang=1"
CSV_DIR = "data/"
os.makedirs(CSV_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Origin": "https://www.pib.gov.in",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# --- Helper Functions ---
def clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace('\xa0', ' ')
    return " ".join(text.split())

def get_safe_filename(name: str) -> str:
    """
    Removes illegal filesystem characters from a string to ensure safe file creation
    while preserving the full name and standard punctuation like '&'.
    """
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def find_select_name(soup: BeautifulSoup, keyword: str) -> str:
    for select in soup.find_all('select'):
        name = select.get('name', '')
        if keyword.lower() in name.lower():
            return name
    return ""

def get_option_value(soup: BeautifulSoup, select_name: str, search_text: str) -> str:
    if not select_name: return ""
    select = soup.find('select', {'name': select_name})
    if not select: return ""
    
    search_lower = search_text.strip().lower()
    search_stripped = search_lower.lstrip('0')
    
    for option in select.find_all('option'):
        val = option.get('value', '').strip().lower()
        text = option.text.strip().lower()
        
        if search_lower in text or search_lower == val or search_stripped == val:
            return option.get('value', '')
            
    first_option = select.find('option')
    return first_option.get('value', '') if first_option else ""

def extract_article_description(html: str) -> str:
    soup = BeautifulSoup(html, 'lxml')
    content_container = soup.find("form", {"name": "aspnetForm"}) or soup.body
    
    if not content_container: return ""

    for tag in content_container(["script", "style", "noscript", "footer"]):
        tag.extract()

    text_elements = content_container.find_all(['p', 'li'])
    extracted_lines = []
    
    for element in text_elements:
        text = clean_text(element.text)
        if text and len(text) > 20: 
            extracted_lines.append(text)

    return "\n".join(extracted_lines)


# --- Core Scraping Modules ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
)
async def fetch_links_for_date(client: httpx.AsyncClient, ministry: str, day: str, month: str, year: str) -> list:
    logger.info(f"Fetching base state for {day}-{month}-{year}...")
    response = await client.get(BASE_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'lxml')
    
    form = soup.find('form')
    if not form: raise ValueError("Could not find HTML form on the page.")
    action_url = urljoin(str(response.url), form.get('action', ''))
    
    payload = {}
    for input_tag in form.find_all('input'):
        name = input_tag.get('name')
        if not name or input_tag.get('type', '').lower() in ['submit', 'button', 'image']:
            continue
        payload[name] = input_tag.get('value', '')
        
    for select_tag in form.find_all('select'):
        name = select_tag.get('name')
        if not name: continue
        selected_option = select_tag.find('option', selected=True)
        payload[name] = selected_option.get('value', '') if selected_option else (select_tag.find('option').get('value', '') if select_tag.find('option') else '')

    ministry_name = find_select_name(soup, "Ministry")
    day_name = find_select_name(soup, "day")
    month_name = find_select_name(soup, "Month")
    year_name = find_select_name(soup, "Year")

    if ministry_name: payload[ministry_name] = get_option_value(soup, ministry_name, ministry)
    if day_name: payload[day_name] = get_option_value(soup, day_name, day)
    if month_name: payload[month_name] = get_option_value(soup, month_name, month)
    if year_name: payload[year_name] = get_option_value(soup, year_name, year)

    submit_btn = form.find('input', type='submit')
    if submit_btn and submit_btn.get('name'):
        payload[submit_btn.get('name')] = submit_btn.get('value', 'Submit')
    else:
        payload["ctl00$ContentPlaceHolder1$btnSubmit"] = "Submit"
        
    for hidden in ["__EVENTTARGET", "__EVENTARGUMENT", "__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION"]:
        if hidden not in payload:
            tag = soup.find("input", {"name": hidden}) or soup.find("input", {"id": hidden})
            payload[hidden] = tag.get('value', '') if tag else ''

    post_headers = dict(HEADERS)
    post_headers["Referer"] = str(response.url)
    
    logger.info(f"Submitting filters for {day}-{month}-{year}...")
    post_response = await client.post(action_url, data=payload, headers=post_headers)
    post_response.raise_for_status()
    
    results_soup = BeautifulSoup(post_response.text, 'lxml')
    if "ErrorPage" in str(post_response.url) or "An error occurred" in results_soup.text:
         raise ValueError(f"EventValidation failed for date {day}-{month}-{year}")

    scraped_data = []
    seen_links = set()

    for a_tag in results_soup.find_all('a', href=True):
        href = a_tag['href']
        text = clean_text(a_tag.text)
        
        if ("PRID=" in href or "PressRelease" in href) and text:
            absolute_url = urljoin(BASE_URL, href)
            if absolute_url not in seen_links:
                seen_links.add(absolute_url)
                scraped_data.append({
                    "heading": text,
                    "link": absolute_url,
                    "description": "" # Empty by default, populated later
                })
                
    return scraped_data

async def populate_descriptions(data_list: list, client: httpx.AsyncClient):
    if not data_list: return data_list
    
    logger.info(f"Fetching descriptions for {len(data_list)} articles concurrently...")
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_single(item: dict):
        async with semaphore:
            try:
                resp = await client.get(item["link"])
                resp.raise_for_status()
                item["description"] = extract_article_description(resp.text)
            except Exception as e:
                logger.error(f"Failed description for {item['link']}: {e}")
                item["description"] = "Failed to fetch description."

    tasks = [fetch_single(item) for item in data_list]
    await asyncio.gather(*tasks)
    return data_list

def save_to_csv(data: list, filename: str) -> str:
    csv_path = os.path.join(CSV_DIR, filename)
    if data:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["heading", "link", "description"])
            writer.writeheader()
            writer.writerows(data)
    return csv_path


# --- Endpoints ---
@app.get("/scrape")
async def scrape_pib_single(
    ministry: str = Query(..., description="Name of the Ministry"),
    day: str = Query(..., description="Day (e.g., 24)"),
    month: str = Query(..., description="Month as digit (e.g., 05 for May)"),
    year: str = Query(..., description="Year (e.g., 2026)")
):
    try:
        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout, headers=HEADERS, verify=False, follow_redirects=True) as client:
            data = await fetch_links_for_date(client, ministry, day, month, year)
            data = await populate_descriptions(data, client)
            
        safe_ministry = get_safe_filename(ministry)
        month_abbr = calendar.month_abbr[int(month)]
        
        # Generates: "Ministry of Agriculture & Farmers Welfare - 24 Jan 2026.csv"
        filename = f"{safe_ministry} - {day} {month_abbr} {year}.csv"
        csv_path = save_to_csv(data, filename)
                
        return JSONResponse(content={
            "status": "success",
            "total_records": len(data),
            "csv_file": csv_path if data else None,
            "data": data
        })

    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/scrape-range")
async def scrape_pib_range(
    ministry: str = Query(..., description="Name of the Ministry (e.g., Prime Minister's Office)"),
    start_date: str = Query(..., description="Start date in DD-MM-YYYY format (e.g. 01-05-2026)"),
    end_date: str = Query(..., description="End' date in DD-MM-YYYY format (e.g. 10-05-2026)")
):
    try:
        start_dt = datetime.strptime(start_date, "%d-%m-%Y")
        end_dt = datetime.strptime(end_date, "%d-%m-%Y")
        
        if start_dt > end_dt:
            raise ValueError("start_date cannot be after end_date")

        timeout = httpx.Timeout(120.0)
        safe_ministry = get_safe_filename(ministry)
        
        all_scraped_data = [] 
        saved_files = []      
        
        current_dt = start_dt
        processing_month = current_dt.month
        processing_year = current_dt.year
        
        monthly_data = []
        seen_monthly_links = set()
        
        # Enclose the processing loop in a try...finally block to guarantee emergency memory dumping
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=HEADERS, verify=False, follow_redirects=True) as client:
                
                while current_dt <= end_dt:
                    
                    if current_dt.month != processing_month or current_dt.year != processing_year:
                        if monthly_data:
                            logger.info(f"Month changed. Processing descriptions for {processing_month}/{processing_year}")
                            
                            monthly_data = await populate_descriptions(monthly_data, client)
                            
                            month_abbr = calendar.month_abbr[processing_month]
                            filename = f"{safe_ministry} - {month_abbr} {processing_year}.csv"
                            
                            csv_path = save_to_csv(monthly_data, filename)
                            saved_files.append(csv_path)
                            all_scraped_data.extend(monthly_data)
                        
                        monthly_data = []
                        seen_monthly_links = set()
                        processing_month = current_dt.month
                        processing_year = current_dt.year

                    day_str = current_dt.strftime("%d")
                    month_str = current_dt.strftime("%m")
                    year_str = current_dt.strftime("%Y")
                    
                    try:
                        daily_data = await fetch_links_for_date(client, ministry, day_str, month_str, year_str)
                        
                        for item in daily_data:
                            if item["link"] not in seen_monthly_links:
                                seen_monthly_links.add(item["link"])
                                monthly_data.append(item)
                    except Exception as e:
                        logger.warning(f"Failed to fetch links for {day_str}-{month_str}-{year_str}: {e}")
                    
                    current_dt += timedelta(days=1)
                
                # Flush the final month after standard operation completes
                if monthly_data:
                    logger.info(f"Processing remaining descriptions for {processing_month}/{processing_year}")
                    monthly_data = await populate_descriptions(monthly_data, client)
                    
                    month_abbr = calendar.month_abbr[processing_month]
                    filename = f"{safe_ministry} - {month_abbr} {processing_year}.csv"
                    
                    csv_path = save_to_csv(monthly_data, filename)
                    saved_files.append(csv_path)
                    all_scraped_data.extend(monthly_data)
                    monthly_data = [] # Clear buffer so the `finally` block knows it's complete

        except asyncio.CancelledError:
            logger.warning("Client disconnected or operation cancelled. Executing emergency save...")
            raise # Re-raises to allow FastAPI to cleanly sever the connection
            
        finally:
            # Emergency Dump: Triggers natively if the try block is forcibly broken by an error or manual cancellation
            if monthly_data:
                logger.info(f"Emergency save for {len(monthly_data)} remaining incomplete records...")
                month_abbr = calendar.month_abbr[processing_month]
                
                # Adds "_PARTIAL" to the filename to denote that this specific file was interrupted mid-process
                filename = f"{safe_ministry} - {month_abbr} {processing_year}_PARTIAL.csv"
                
                csv_path = save_to_csv(monthly_data, filename)
                saved_files.append(csv_path)
                all_scraped_data.extend(monthly_data)
                
        return JSONResponse(content={
            "status": "success",
            "total_records": len(all_scraped_data),
            "csv_files": saved_files,
            "data": all_scraped_data
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Range Scraping critically failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pib_scrapper:app", host="0.0.0.0", port=8000, reload=True)
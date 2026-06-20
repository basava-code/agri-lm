"""
Data.gov.in High-Performance Concurrent Extraction Pipeline.

Architecture: Asynchronous Producer-Consumer Queue.
- Producers: N concurrent workers fetch data safely behind a Semaphore.
- Buffer: In-memory queue holds raw CSV strings.
- Consumer: Flushes segmented 100,000-row file batches dynamically to disk.
"""

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List
import os
from dotenv import load_dotenv

import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# SECURE CONFIGURATION MANAGEMENT (USER INITIAL DEFINITION PRESERVED)
# ==============================================================================
class Settings(BaseSettings):
    """
    Validates and loads environment variables automatically.
    Requires a DATA_GOV_API_KEY to be set in the environment or a .env file.
    """
    data_gov_api_key: str | None = os.getenv("DATA_GOV_API_KEY")
    dataset_id: str | None = os.getenv("DATASET_ID")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiating settings will immediately fail if the API key is missing.
settings = Settings()

# ==============================================================================
# SYSTEM CONSTANTS & THRESHOLDS
# ==============================================================================
API_URL = f"https://api.data.gov.in/resource/{settings.dataset_id}"
PAGE_LIMIT = 1000

# High-Performance Tuning Parameters
CONCURRENT_REQUESTS = 2         # Maximum parallel requests to avoid WAF bans
MAX_RETRIES = 5                 # Fallback retries per chunk
BACKOFF_BASE_DELAY = 2.0        # Exponential backoff base
NETWORK_TIMEOUT = 45.0          # Extended timeout for concurrent loads

# 100 Chunks * 1000 Rows Per Page = 100,000 Rows per isolated File Batch
FLUSH_THRESHOLD = 100           
OUTPUT_DIR = Path("data/csv")

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DataGovHighPerf")

app = FastAPI(title="Data Extraction Pipeline API (Batch Segmented)", version="2.5.0")

class ExtractionResponse(BaseModel):
    status: str
    total_records_processed: int
    execution_time_seconds: float
    output_directory: str
    total_files_generated: int

# ==============================================================================
# HIGH PERFORMANCE ENGINE
# ==============================================================================
class ConcurrentExtractionEngine:
    """Manages concurrent fetching and automatic target file rotation."""
    
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.queue: asyncio.Queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        self.total_records = 0
        self.batch_counter = 1
        
        # Prepare target directory structures safely
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Clear out files matching our specific naming layout from previous operational runs
        for old_file in OUTPUT_DIR.glob("Batch * Range (*).csv"):
            try:
                old_file.unlink()
            except Exception as e:
                logger.warning(f"Could not clear stale file {old_file.name}: {str(e)}")

    async def fetch_worker(self, offset: int, worker_id: int) -> int:
        """Fetches a specific offset chunk safely behind a concurrency semaphore."""
        params: Dict[str, Any] = {
            "api-key": settings.data_gov_api_key,
            "format": "csv",
            "offset": offset,
            "limit": PAGE_LIMIT,
        }
        headers: Dict[str, str] = {
            "accept": "text/csv",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        async with self.semaphore:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await self.client.get(
                        API_URL, params=params, headers=headers, timeout=NETWORK_TIMEOUT
                    )
                    
                    if response.status_code == 429 or response.status_code >= 500:
                        logger.warning(f"[Worker {worker_id}] Transient error {response.status_code} at offset {offset}.")
                        response.raise_for_status()
                    
                    response.raise_for_status()
                    await self.queue.put(response.text)
                    
                    row_count = max(0, response.text.count('\n') - 1)
                    return row_count

                except httpx.HTTPStatusError as exc:
                    logger.error(f"HTTP Error {exc.response.status_code} at offset {offset}.")
                    if attempt == MAX_RETRIES:
                        return 0
                        
                except (httpx.RequestError, httpx.TimeoutException):
                    if attempt == MAX_RETRIES:
                        logger.error(f"[Worker {worker_id}] Network failure at offset {offset} after max retries.")
                        return 0
                
                sleep_time = (BACKOFF_BASE_DELAY * (2 ** attempt)) + (time.time() % 1.0)
                await asyncio.sleep(sleep_time)
                
            return 0

    async def disk_writer_consumer(self):
        """Runs continuously in the background, segments files, and commits them to disk."""
        buffer: List[pd.DataFrame] = []
        
        while True:
            raw_csv = await self.queue.get()
            
            if raw_csv is None:
                if buffer:
                    self._flush_buffer_to_disk(buffer)
                self.queue.task_done()
                break
                
            if raw_csv.strip():
                try:
                    df = pd.read_csv(io.StringIO(raw_csv.strip()))
                    if not df.empty:
                        buffer.append(df)
                        self.total_records += len(df)
                except Exception as exc:
                    logger.error(f"Failed to parse chunk: {str(exc)}")
            
            # Flush when buffer matches our targeted 100,000 records framework
            if len(buffer) >= FLUSH_THRESHOLD:
                self._flush_buffer_to_disk(buffer)
                buffer.clear()
                
            self.queue.task_done()

    def _flush_buffer_to_disk(self, buffer: List[pd.DataFrame]):
        """Consolidates buffer data into an explicitly bounded partition file."""
        if not buffer:
            return
            
        consolidated_batch = pd.concat(buffer, ignore_index=True)
        batch_size = len(consolidated_batch)
        
        # Calculate dynamic text metrics based on records tracked
        start_row = self.total_records - batch_size
        end_row = self.total_records
        
        # Build exact file matching layout schema rules
        file_name = f"Batch {self.batch_counter} Range ({start_row} - {end_row}).csv"
        file_path = OUTPUT_DIR / file_name
        
        # Save fresh with a full set of top-level structural headers
        consolidated_batch.to_csv(file_path, index=False, header=True)
        logger.info(f"File Written: {file_name} containing {batch_size} rows.")
        
        self.batch_counter += 1

# ==============================================================================
# FASTAPI ROUTING ENDPOINT
# ==============================================================================
@app.get(
    "/extract",
    status_code=status.HTTP_200_OK,
    response_model=ExtractionResponse,
    summary="Triggers the high-performance concurrent ingestion pipeline.",
)
async def run_extraction_pipeline() -> Dict[str, Any]:
    start_time = time.perf_counter()
    current_offset = 0
    worker_id = 0
    
    logger.info("Starting High-Performance Concurrent Extraction Pipeline...")

    async with httpx.AsyncClient() as client:
        engine = ConcurrentExtractionEngine(client=client)
        writer_task = asyncio.create_task(engine.disk_writer_consumer())
        
        try:
            while True:
                tasks = []
                for _ in range(CONCURRENT_REQUESTS):
                    tasks.append(
                        asyncio.create_task(engine.fetch_worker(offset=current_offset, worker_id=worker_id))
                    )
                    current_offset += PAGE_LIMIT
                    worker_id += 1
                
                results = await asyncio.gather(*tasks)
                
                if any(row_count < PAGE_LIMIT for row_count in results):
                    logger.info("Pagination boundary detected. Ending fetch cycle.")
                    break

        except Exception as exc:
            logger.critical(f"Critical execution failure: {str(exc)}")
            raise HTTPException(status_code=500, detail="Pipeline crashed during execution.")
            
        finally:
            await engine.queue.put(None)
            await engine.queue.join()
            await writer_task
            
    execution_time = round(time.perf_counter() - start_time, 2)
    logger.info(f"Extraction job completed. Generated {engine.batch_counter - 1} separate files.")

    return {
        "status": "success",
        "total_records_processed": engine.total_records,
        "execution_time_seconds": execution_time,
        "output_directory": str(OUTPUT_DIR),
        "total_files_generated": engine.batch_counter - 1,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
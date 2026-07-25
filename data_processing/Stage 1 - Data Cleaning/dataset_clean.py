import os
import re
import shutil
import tempfile
import logging
import unicodedata
from typing import List

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask, BackgroundTasks

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants (Added KccAns)
REQUIRED_COLS = [
    "BlockName", "Category", "CreatedOn", "Crop", "DistrictName",
    "KccAns", "QueryText", "QueryType", "Sector", "StateName", "day", "month", "year"
]

# Output Constants (Added Answer after Question)
OUTPUT_COLS = [
    "Id", "Question", "Answer", "Crop", "Category", "Sector",
    "Date & Time", "Location"
]

# Initialize FastAPI App
app = FastAPI(
    title="CSV ETL Pipeline API",
    description="Production-grade API to clean, transform, and deduplicate raw CSV datasets.",
    version="1.0.0"
)


def cleanup_files(*filepaths: str) -> None:
    """Safely delete temporary files."""
    for fp in filepaths:
        try:
            if os.path.exists(fp):
                os.remove(fp)
                logger.debug(f"Successfully deleted temp file: {fp}")
        except Exception as e:
            logger.error(f"Failed to delete temp file {fp}: {str(e)}")


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFC format."""
    if not text:
        return text
    return unicodedata.normalize('NFC', text)


def to_sentence_case(text: str) -> str:
    """Convert text to properly formatted English sentences (standard sentence case)."""
    if not text:
        return text
    
    # Lowercase the entire string to establish a baseline
    text = text.lower()
    
    # Capitalize the first letter of the string and the first letter following end punctuation
    # Regex breakdown:
    # (?:^|[.!?]\s+) matches the start of the string OR a punctuation mark followed by whitespace(s)
    # ([a-z]) matches the first alphabetical character to be capitalized
    return re.sub(r'(?:^|[.!?]\s+)([a-z])', lambda m: m.group(0).upper(), text)


def clean_series(s: pd.Series) -> pd.Series:
    """Apply robust vector-based text cleaning and formatting to a pandas Series."""
    # Handle NaN, NULL, empty values by converting to empty string
    s = s.fillna("").astype(str)
    
    # Strip newline characters and tabs
    s = s.str.replace(r'[\r\n\t]+', ' ', regex=True)
    
    # Replace multiple spaces with a single space
    s = s.str.replace(r'\s{2,}', ' ', regex=True)
    
    # Trim leading and trailing whitespace
    s = s.str.strip()
    
    # Apply unicode normalization
    s = s.apply(normalize_unicode)
    
    # Apply standard sentence case to every column
    s = s.apply(to_sentence_case)
    
    return s


def parse_dates(df: pd.DataFrame) -> pd.Series:
    """Parse dates from CreatedOn or fallback to day, month, year."""
    # Attempt parsing CreatedOn (preferring dayfirst due to typical Indian formats)
    dt_created = pd.to_datetime(df['CreatedOn'], errors='coerce', dayfirst=True)

    # Safely construct dates from year, month, day columns
    dmy_df = pd.DataFrame({
        'year': pd.to_numeric(df['year'], errors='coerce'),
        'month': pd.to_numeric(df['month'], errors='coerce'),
        'day': pd.to_numeric(df['day'], errors='coerce')
    })
    dt_dmy = pd.to_datetime(dmy_df, errors='coerce')

    # Combine: Prefer CreatedOn, fallback to constructed Date
    dt_final = dt_created.combine_first(dt_dmy)

    # Format as ISO-like string and fill NaT with blank values
    return dt_final.dt.strftime('%Y-%m-%d %H:%M:%S').fillna("")


def construct_location(df: pd.DataFrame) -> pd.Series:
    """Construct location string vectorially from Block, District, and State."""
    b = df['BlockName']
    d = df['DistrictName']
    s = df['StateName']

    # Vectorized masks to check presence of trailing components
    b_has_more = (b != '') & ((d != '') | (s != ''))
    b_str = b + np.where(b_has_more, ', ', '')

    d_has_more = (d != '') & (s != '')
    d_str = d + np.where(d_has_more, ', ', '')

    # Combine into single location string
    return b_str + d_str + s


def process_csv(input_path: str, output_path: str) -> None:
    """Core ETL pipeline to read, transform, deduplicate, and save the CSV."""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    df = None

    # Step 1: Read CSV with robust encoding handling
    for enc in encodings:
        try:
            # Validate required columns strictly first without loading full dataset
            header = pd.read_csv(input_path, nrows=0, encoding=enc)
            missing_cols = [c for c in REQUIRED_COLS if c not in header.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

            # Read only required columns to save memory, handle corrupted lines gracefully
            df = pd.read_csv(
                input_path,
                encoding=enc,
                usecols=REQUIRED_COLS,
                dtype=str,
                on_bad_lines='skip'
            )
            break
        except ValueError as ve:
            if "Missing required columns" in str(ve):
                raise
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded CSV file is empty.")
        except pd.errors.ParserError:
            continue

    if df is None:
        raise ValueError("Failed to read CSV. It might be corrupted or using an unsupported encoding.")

    # Step 2: Data Cleaning and Formatting
    for col in REQUIRED_COLS:
        df[col] = clean_series(df[col])
        
    # Step 3: Deduplicate rows based on the cleaned 'QueryText' (Question)
    # Keeping the first occurrence and dropping the rest, then reset index for seamless ID generation
    df = df.drop_duplicates(subset=["QueryText"], keep="first").reset_index(drop=True)

    # Step 4: Transformation & Column Mapping
    df_out = pd.DataFrame()
    
    df_out["Id"] = np.arange(1, len(df) + 1)
    df_out["Question"] = df["QueryText"]
    df_out["Answer"] = df["KccAns"]  # Added the Answer Column mapping here
    df_out["Crop"] = df["Crop"]
    df_out["Category"] = df["Category"]
    df_out["Sector"] = df["Sector"]
    df_out["Date & Time"] = parse_dates(df)
    df_out["Location"] = construct_location(df)

    # Step 5: Output writing (Ensuring correct column order & UTF-8 encoding)
    df_out = df_out[OUTPUT_COLS]
    df_out.to_csv(output_path, index=False, encoding='utf-8')


@app.post("/clean")
async def clean_csv_endpoint(file: UploadFile = File(...)):
    """API endpoint to upload a raw CSV and download the cleaned version."""
    
    if not file.filename.lower().endswith('.csv'):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid file type. Only CSV files are allowed."}
        )

    # Use secure temporary files for processing
    fd_in, temp_in_path = tempfile.mkstemp(suffix=".csv")
    fd_out, temp_out_path = tempfile.mkstemp(suffix=".csv")
    
    try:
        # Stream the uploaded file to disk securely
        with os.fdopen(fd_in, "wb") as f_in:
            shutil.copyfileobj(file.file, f_in)
            
        # Check if the file is genuinely empty
        if os.path.getsize(temp_in_path) == 0:
            raise ValueError("The uploaded CSV file is empty.")

        # Execute ETL Pipeline
        process_csv(temp_in_path, temp_out_path)

        # Prepare Background Task to clean up temp files after response is fully sent
        bg_tasks = BackgroundTasks()
        bg_tasks.add_task(cleanup_files, temp_in_path, temp_out_path)

        return FileResponse(
            path=temp_out_path,
            filename=f"cleaned_{file.filename}",
            media_type="text/csv",
            background=bg_tasks
        )

    except ValueError as ve:
        # Catch our custom validations and pandas errors
        cleanup_files(temp_in_path, temp_out_path)
        logger.warning(f"Validation Error: {str(ve)}")
        return JSONResponse(status_code=400, content={"detail": str(ve)})
        
    except Exception as e:
        # Catch unexpected pipeline/system crashes
        cleanup_files(temp_in_path, temp_out_path)
        logger.exception("Unexpected error during CSV processing")
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred during processing."}
        )
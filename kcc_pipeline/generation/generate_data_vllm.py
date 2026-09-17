import os
import glob
import pandas as pd
from openai import OpenAI
import json
import concurrent.futures
from tqdm import tqdm
import time

# vLLM Server Configuration
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "/home/basava/agri-lm/kcc-data-cleaning/gemma-4-31b-it"

INPUT_DIR = "raw_data"
OUTPUT_DIR = "prepared_data"
PROGRESS_FILE = "pipeline_state.json"

BATCH_SIZE = 10
MAX_WORKERS = 45 # Number of concurrent requests sent to the vLLM server

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="vllm-does-not-care",
    timeout=None
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_state():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def generate_cpt_text(df_batch):
    records = df_batch[['QueryText', 'KccAns']].to_dict(orient='records')
    records_str = json.dumps(records, ensure_ascii=False)

    prompt = f"""
    You are an expert Agricultural AI Dataset Creator.
    Below is a JSON array of {len(records)} raw logs from an Indian Farmer Call Center.
    The text is highly multilingual and contains pure native language scripts (Devanagari/Hindi, Gurmukhi/Punjabi, Tamil, Kannada, Marathi, Malayalam, etc.) mixed with Hinglish and English.
    
    Your Task:
    1. Read the farmer queries and expert answers.
    2. Extract ONLY the factual agricultural knowledge.
    3. Translate these facts into high-quality, professional English.
    4. Synthesize the extracted facts into a contiguous "Textbook" or "Wiki-style" format suitable for Continual Pre-Training of a Language Model.

    Strict Rules:
    - DO NOT include conversational filler, greetings, or administrative complaints.
    - DO NOT hallucinate. Only include facts present in the provided JSON.
    - Output ONLY the synthesized educational text. Do not include markdown blocks or JSON.
    - If the batch contains absolutely no useful agricultural facts, output EXACTLY the word "SKIP".

    Input Data:
    {records_str}
    """
    query = f"""
    Array Length: {len(records)}

    Input Data:
    {records_str}
    """

    
    while True:
        try:
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{'role': 'system', 'content': prompt}, {'role': 'user', 'content': query}],
                max_tokens=2500,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "400" in error_msg or "context length" in error_msg:
                print(f"\nCorrupted massive row detected (Exceeds 12k tokens). Skipping batch to save pipeline.")
                return "<SKIPPED_CHUNK>"
            print(f"\nConnection to vLLM failed: {e}. Retrying in 10 seconds...")
            time.sleep(10)
            

def format_chunk(synthesized_text, start_index):
    end_index = start_index + BATCH_SIZE -1
    header = f"=== BATCH_ROWS_{start_index}_TO_{end_index} ==="
    if synthesized_text and "SKIP" not in synthesized_text.upper():
        return f"{header}\n{synthesized_text}\n<eos>\n\n"
    else:
        return f"{header}\n<SKIPPED_CHUNK>\n<eos>\n\n"

def process_datasets():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}/")
        return

    print(f"Starting VLLM Generation Pipeline using {MAX_WORKERS} concurrent workers...")
    
    state = load_state()

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        output_filename = filename.replace('.csv', '_cpt.txt')
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if filename in state and state[filename].get("status") == "completed":
            print(f"Skipping {filename} (Already completed).")
            continue

        print(f"\nProcessing {filename} -> {output_filename}")
        
        df = pd.read_csv(file_path, low_memory=False)
        df = df.dropna(subset=['QueryText', 'KccAns'])
        
        total_rows = len(df)
        total_batches = (total_rows // BATCH_SIZE) + (1 if total_rows % BATCH_SIZE != 0 else 0)
        
        if filename not in state:
            state[filename] = {"last_processed_index": 0, "status": "in_progress"}
            file_mode = 'w'  
        else:
            file_mode = 'a'  
            print(f"Resuming {filename} from row {state[filename]['last_processed_index']}...")

        start_index = state[filename]["last_processed_index"]
        
        pending_indices = list(range(start_index, total_rows, BATCH_SIZE))
        if not pending_indices:
            state[filename]["status"] = "completed"
            save_state(state)
            continue

        with open(output_path, file_mode, encoding='utf-8') as out_file:
            
            # Since threads complete out of order, we MUST buffer results
            # so we write them sequentially to the file and preserve the Judge Agent's alignment!
            buffered_results = {}
            next_write_index = start_index

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Fire off all API requests concurrently
                future_to_index = {
                    executor.submit(generate_cpt_text, df.iloc[i : i + BATCH_SIZE]): i 
                    for i in pending_indices
                }

                pbar = tqdm(total=total_batches, initial=start_index // BATCH_SIZE, desc="Batches")

                for future in concurrent.futures.as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        synthesized_text = future.result()
                    except Exception as exc:
                        print(f"Chunk starting at {i} generated an exception: {exc}")
                        synthesized_text = "SKIP"
                    
                    buffered_results[i] = format_chunk(synthesized_text, i)

                    # Check if the NEXT expected sequential block is ready to be flushed to disk
                    while next_write_index in buffered_results:
                        clean_chunk = buffered_results.pop(next_write_index)
                        out_file.write(clean_chunk)
                        out_file.flush()
                        
                        next_write_index += BATCH_SIZE
                        state[filename]["last_processed_index"] = next_write_index
                        save_state(state)
                        
                    pbar.update(1)

        state[filename]["status"] = "completed"
        save_state(state)
        print(f"Finished processing {filename}!")

if __name__ == "__main__":
    try:
        process_datasets()
        print("\nPipeline Complete! Your CPT Dataset is ready in prepared_data/")
    except KeyboardInterrupt:
        print("\nPipeline paused safely. It will resume exactly from this spot on next run.")

# Book Data Extraction Pipeline Architecture

This document explains the architecture and design decisions behind the QnA dataset generation pipeline.

## 1. Generic PDF Chapter Chunker (`chunker.py`)

### Total Independence
By separating the TOC alignment from the PDF slicing, the chunker is completely book-agnostic. It does not contain any agricultural keywords, variety names, or printed ranges. 

### Dynamic JSON Loading
It accepts a `--toc` command-line argument pointing to a dynamically resolved TOC JSON file. It parses the keys and slices the PDF based on the custom ranges provided.

### Backward Compatibility
To ensure the downstream QnA batch spawning and dataset formatting scripts function without modifying their inputs, the chunker saves the output to a specific book-named path.

### Text Cleaning & Noise Reduction
PDF files are formatted for visual presentation, not semantic extraction. When extracting text, we capture "background noise" such as running page headers (e.g., 'AGRICULTURAL MANUAL') and isolated page numbers. We dynamically filter out lines that contain only digit strings or match known headers to avoid introducing random noise into our LLM context.

Additionally, in PDFs, a single word is often split across two lines with a hyphen (e.g., 'trans-\nplanting'). If we send this directly to the LLM, the tokeniser treats 'trans-' and 'planting' as separate tokens, hurting vocabulary lookup. A regular expression is used to stitch word-hyphens followed by a line break back together.

## 2. Converting to CSV (`convert_to_csv.py`)

- **Robust Path Handling**: `pathlib.Path` is used for OS-independent path handling, making the script robust whether run from the repository root or a sub-directory.
- **Explicit Role Checks**: The JSONL format may contain additional system messages. We extract only the first `user` and the first `assistant` messages to form a clean QnA pair.
- **Error Handling**: Malformed lines are intentionally skipped but logged to `stderr` so users can inspect problematic entries without aborting the script entirely.

## 3. Dataset Formatting (`dataset_formatter.py`)

This stage unpacks the offline, asynchronous Batch API (or local async generator) responses:

### Unpacking the JSONL Structure
The output is a JSON Lines (.jsonl) file where each object contains a 'key' (unique request ID) and a 'response'. The file is read line-by-line to check for errors and extract content candidates.

### Extracting Structured Outputs
Because we use Pydantic Structured Outputs in the prompts, the LLM strictly serializes text matching our schema. We parse the string to get our typed dictionary of QnA groups.

### Flattening Variations
Each `FactQnAGroup` contains multiple variations of asking/answering the same underlying fact. To train our model, we 'flatten' these variations, writing each variation as a standalone training sample. This increases dataset diversity and teaches the model to respond consistently to multiple question formulations.

### Formatter Output (Chat Template)
Modern instructional models like Gemma are trained on conversational chat templates:
`{"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}`
This matches the Hugging Face standard and makes data ingestion instantaneous. The model is trained on the 'assistant' turns by auto-regressively masking the 'user' content and computing loss solely on the target assistant text.

# Dual-Engine Agricultural Model Evaluation & Edge GGUF Export

Automated multi-stage evaluation harness and edge quantization pipeline.

## Dual-Engine Evaluation Architecture (`eval_cpt_ollama_vllm.py`)
- **Examiner:** `Gemma-4-31B-FP8` running on an OpenAI-compatible vLLM server.
- **Candidate:** Domain-adapted candidate model served via Ollama or local endpoint.
- **Evaluation Tiers:**
  1. **Stage 1 (Chapter-Level Q&A):** Assesses factual retention, quantitative dosage precision, and soil chemical thresholds.
  2. **Stage 2 (Multi-Turn Examination):** Deep reasoning probe examining diagnostic consistency under noisy farmer descriptions.
  3. **Stage 3 (Whole-Book Macro Synthesis):** Global pedological comprehension and cross-chapter synthesis.

## GGUF Quantization & Multi-Modal Export (`export_gguf.py`)
- Fuses trained LoRA adapters into base model weights.
- Emits FP16, BF16, and Q4_K_M GGUF binaries for CPU/Edge llama.cpp inference.
- Quantizes paired vision projectors (`-mmproj.gguf`) for multi-modal VLM deployments.

## Usage

```bash
# Run dual-engine evaluation
python eval_cpt_ollama_vllm.py --model gemma4_e2b_acid_soils_v3.2:f16

# Export LoRA to GGUF
python export_gguf.py --model-dir /path/to/checkpoint --output-dir ./gguf/ --quantization Q4_K_M
```

# 🧠 Sovereign Agri-LM Fine-Tuning Suite

[![Unsloth](https://img.shields.io/badge/FineTuned_With-Unsloth-2ea44f.svg)](https://github.com/unslothai/unsloth)
[![LoRA](https://img.shields.io/badge/Method-LoRA_&_rsLoRA-purple.svg)](https://arxiv.org/abs/2106.09685)
[![Precision](https://img.shields.io/badge/Precision-Native_BF16_/_16--bit_LoRA-blue.svg)](https://pytorch.org/)

The **Agri-LM Fine-Tuning Suite** implements a disciplined two-phase domain adaptation protocol to instill dense pedological facts, chemical dosages, and regional agronomic advisory into foundation models without suffering from catastrophic forgetting.

---

## 🎯 The Two-Phase Sovereign Adaptation Strategy

```
Phase 1: Continuous Pre-Training (CPT)              Phase 2: Supervised Fine-Tuning (SFT)
=====================================              ====================================
• Factual Knowledge Infusion                       • Conversational & Diagnostic Alignment
• Unsloth 16-bit LoRA on ALL 7 Projections         • Response-Only Loss Masking (train_on_responses_only)
• True Manual Sequence Packing (4096 Tokens)       • Cross-Entropy Loss strictly on Assistant Turns
• Targets Feed-Forward Networks (FFNs)             • Zero Loss Computed on User Prompts
```

### 🔬 Why FFN LoRA is Mandatory for Factual Recall
Standard fine-tuning often adapts only attention query and value matrices (`q_proj`, `v_proj`). Empirical research in transformer memory localization demonstrates that **encyclopedic knowledge and factual associative memory reside predominantly inside Feed-Forward Networks (FFNs)**: `gate_proj`, `up_proj`, and `down_proj`. Agri-LM adapts all 7 linear projections simultaneously:
$$\{q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj\}$$

---

## 🛠️ Phase 1: Continual Pre-Training (`unsloth_cpt_trainer.py`)

### 1. Multimodal Manual Sequence Packing
Hugging Face TRL natively disables sequence packing for vision-language models (such as `Gemma-4-E2B-it`). Naive training without packing pads batches with `<pad>` tokens, wasting up to 70% of GPU compute and memory bandwidth.

`unsloth_cpt_trainer.py` solves this via `manually_pack_dataset()`, dynamically concatenating documents into dense 4,096-token blocks separated by boundary tokens:
$$\langle bos \rangle \text{Doc}_1 \langle eos \rangle \text{Doc}_2 \langle eos \rangle \dots \langle eos \rangle$$

### 2. Execution Command
```bash
python unsloth_cpt_trainer.py \
    --model_name unsloth/gemma-4-e2b-it \
    --dataset_path "../data/processed/Acid Soils of India/cpt_dataset_packed.jsonl" \
    --output_dir outputs/cpt/gemma4_e2b_cpt_v1 \
    --merged_dir outputs/cpt/gemma4_e2b_cpt_v1_merged \
    --epochs 2 \
    --rank 64 \
    --alpha 128 \
    --learning_rate 3e-5 \
    --max_seq_length 4096
```

---

## 🎯 Phase 2: Response-Masked SFT (`unsloth_sft_trainer.py`)

### 1. Answer Loss Masking (`train_on_responses_only`)
Standard causal language modeling computes loss across the entire token sequence, penalizing the model for predicting common phrasing in the user's prompt. 

Our SFT trainer integrates `train_on_responses_only`:
- User conversational queries and context tokens are **masked with token ID `-100`** during loss computation.
- **100% of gradient updates penalize factual inaccuracies, improper dosages, and wrong diagnoses in the assistant's answer.**

### 2. Execution Command
```bash
python unsloth_sft_trainer.py \
    --model_name outputs/cpt/gemma4_e2b_cpt_v1_merged \
    --dataset_path "../data/processed/Acid Soils of India/acid_soils_sft_dataset.jsonl" \
    --output_dir outputs/sft/gemma4_e2b_sft_v1 \
    --epochs 4 \
    --batch_size 2 \
    --grad_accum 2 \
    --learning_rate 2e-4 \
    --rank 32 \
    --alpha 64 \
    --merge_16bit
```

---

## ⚙️ Declarative YAML Configurations (`configs/`)

For standardized training runs, use the declarative YAML configurations in `configs/`:

- **`configs/cpt_train.yaml`:**
  - Rank: 128, Alpha: 256 (`rsLoRA: true`)
  - Target Modules: All 7 projections
  - Optimizer: `paged_adamw_8bit`
  - Max Context Length: 8,192 tokens with sequence packing enabled
- **`configs/sft_qna_train.yaml`:**
  - Rank: 16, Alpha: 32
  - Target Modules: All 7 projections
  - Optimizer: `paged_adamw_8bit`
  - Max Context Length: 1,024 tokens (packing disabled to prevent cross-example bleed)

---

## 🔀 Standalone 16-bit LoRA Adapter Merging

Once fine-tuning completes, the adapter weights can be fused back into the unquantized base model weights for zero-overhead inference:

```bash
python src/merger.py \
    --base-model unsloth/gemma-4-e2b-it \
    --adapter-path outputs/sft/gemma4_e2b_sft_v1 \
    --output-dir outputs/sft/gemma4_e2b_sft_v1_merged \
    --precision bfloat16
```
The resulting directory contains standard Hugging Face `config.json` and `model.safetensors` files, ready for vLLM, Ollama, or GGUF compilation.

---

## 📊 VRAM & Hardware Footprint

| Phase | Model Architecture | Sequence Length | Batch Size (x Accum) | Peak VRAM | Recommended GPU |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **CPT (Packed)** | Gemma-4-E2B (2.6B) | 4,096 | 2 x 4 | ~18.5 GB | RTX 3090 / 4090 (24 GB) |
| **CPT (Deep)** | Gemma-3-1B / 4-E2B | 8,192 | 2 x 4 | ~22.0 GB | RTX 4090 / A10G (24 GB) |
| **SFT (Masked)** | Gemma-4-E2B (2.6B) | 2,048 | 2 x 2 | ~14.2 GB | RTX 3080 Ti (16 GB+) |
| **Full Precision Merge** | All Variants | N/A | N/A | ~12.0 GB RAM | CPU RAM sufficient |

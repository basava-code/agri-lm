#!/usr/bin/env python3
"""
High-Precision SFT Trainer for Gemma-4 with Answer Loss Masking.
Trains specifically on Q&A responses using train_on_responses_only so that 100% of the gradient
updates penalize factual inaccuracies in the model's answers.
"""

import os
import sys
import argparse
import logging
import json
import torch

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _text_tokenizer(processor_or_tokenizer):
    """Unwraps Gemma4Processor to get the underlying text tokenizer."""
    if hasattr(processor_or_tokenizer, "tokenizer"):
        return processor_or_tokenizer.tokenizer
    return processor_or_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Unsloth SFT Trainer with Answer Loss Masking")
    parser.add_argument("--model_name", type=str,
                        default="outputs/cpt/gemma4_e2b_acid_soils_v3_merged",
                        help="Base or CPT model path")
    parser.add_argument("--dataset_path", type=str,
                        default="data/processed/Acid Soils of India/acid_soils_sft_dataset.jsonl",
                        help="Path to SFT JSONL dataset with messages")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/sft/gemma4_e2b_acid_soils_sft",
                        help="Output directory for SFT checkpoints and merged model")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--merge_16bit", action="store_true", default=True)
    args = parser.parse_args()

    logging.info(f"Loading Model & Tokenizer from: {args.model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )

    logging.info(f"Configuring LoRA: rank={args.rank}, alpha={args.alpha}, dropout=0 (fast patching)")
    model = FastModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    text_tok = _text_tokenizer(tokenizer)
    # Apply standard Google Gemma chat template ("gemma")
    try:
        text_tok = get_chat_template(text_tok, chat_template="gemma")
    except Exception as e:
        logging.warning(f"Using native chat template from model tokenizer ({e})")

    logging.info(f"Loading dataset from: {args.dataset_path}")
    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = []
        for convo in convos:
            formatted_convo = []
            for msg in convo:
                r = msg.get("role", "user")
                c = msg.get("content", "")
                if r in ["assistant", "model"]:
                    formatted_convo.append({"role": "assistant", "content": c})
                else:
                    formatted_convo.append({"role": "user", "content": c})
            text = text_tok.apply_chat_template(formatted_convo, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    formatted_dataset = raw_dataset.map(formatting_prompts_func, batched=True)
    logging.info(f"Loaded {len(formatted_dataset)} formatted SFT training examples.")
    
    sample_text = formatted_dataset[0]["text"]
    logging.info(f"Sample formatted prompt preview:\n---\n{sample_text[:300]}...\n---")

    # Detect exact response marker
    if "<start_of_turn>model\n" in sample_text:
        resp_part = "<start_of_turn>model\n"
        inst_part = "<start_of_turn>user\n"
    elif "<start_of_turn>model" in sample_text:
        resp_part = "<start_of_turn>model"
        inst_part = "<start_of_turn>user"
    elif "<start_of_turn>assistant\n" in sample_text:
        resp_part = "<start_of_turn>assistant\n"
        inst_part = "<start_of_turn>user\n"
    else:
        resp_part = "<start_of_turn>model"
        inst_part = "<start_of_turn>user"

    logging.info(f"Using instruction_part='{inst_part}', response_part='{resp_part}'")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=2,
        save_strategy="epoch",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=text_tok,
        train_dataset=formatted_dataset,
        args=sft_config,
    )

    logging.info("Applying train_on_responses_only loss masking (user instructions will be masked out of loss)...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=inst_part,
        response_part=resp_part,
    )

    logging.info("Starting SFT Training...")
    trainer.train()

    adapter_path = os.path.join(args.output_dir, "final_adapter")
    logging.info(f"Saving final SFT adapter to: {adapter_path}")
    model.save_pretrained(adapter_path)
    text_tok.save_pretrained(adapter_path)

    if args.merge_16bit:
        merged_path = os.path.join(args.output_dir + "_merged")
        logging.info(f"Merging and exporting 16-bit standalone model to: {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer=text_tok, save_method="merged_16bit")
        logging.info(f"Successfully exported 16-bit standalone SFT model: {merged_path}")

    logging.info("SFT Training Complete!")

if __name__ == "__main__":
    main()

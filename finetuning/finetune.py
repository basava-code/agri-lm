import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    # 0. Early GPU Availability Verification
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("ERROR: NO GPU DETECTED!")
        print("It appears your Google Colab session is running on a CPU-only runtime.")
        print("Fine-tuning Gemma-2-2B in 4-bit QLoRA requires a GPU (T4 or higher).")
        print("Please switch your runtime to a GPU by going to:")
        print("   Runtime -> Change runtime type -> Select 'T4 GPU' under Hardware Accelerator.")
        print("="*80 + "\n")
        raise RuntimeError("GPU is required for QLoRA fine-tuning.")
        
    # Detect BF16 support
    supports_bf16 = torch.cuda.is_bf16_supported()
    print(f"Hardware Detection: GPU Available = True | BF16 Supported = {supports_bf16}")

    # ----------------------------------------------------
    # Configuration
    # ----------------------------------------------------
    model_id = "./gemma-2-2b-it-cpt-merged"  # Start SFT from your locally CPT-merged base model!
    dataset_path = "target_book_gemma_qna_dataset_new.jsonl"
    adapter_output_dir = "./gemma-2-2b-it-qna-adapter"
    merged_output_dir = "./gemma-2-2b-it-qna-merged"

    print("--- 1. Loading and Formatting Dataset ---")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please upload it to the current directory.")

    # Load dataset using Hugging Face datasets
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} examples.")

    # Load tokenizer
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Format the dataset using the tokenizer's chat template
    def apply_chat_template(sample):
        # The dataset structure is {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        # SFTTrainer can use a formatting function that returns formatted text strings.
        formatted_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        return {"text": formatted_text}

    formatted_dataset = dataset.map(apply_chat_template, remove_columns=["messages"])
    print("Example formatted text:")
    print(formatted_dataset[0]["text"])

    # ----------------------------------------------------
    # Model Loading (4-bit Quantized)
    # ----------------------------------------------------
    print("\n--- 2. Loading Base Model in 4-bit Quantization ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # ----------------------------------------------------
    # LoRA Config
    # ----------------------------------------------------
    print("\n--- 3. Configuring LoRA (PEFT) ---")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Target all linear layers for best performance
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ----------------------------------------------------
    # SFT Configuration (replaces TrainingArguments in newer TRL versions)
    # ----------------------------------------------------
    print("\n--- 4. Setting up SFT Configuration ---")
    sft_config = SFTConfig(
        output_dir="./results",
        num_train_epochs=3,                  # 3 epochs is usually optimal for QnA fine-tuning
        per_device_train_batch_size=2,       # Low batch size to prevent OOM on T4 GPU (16GB VRAM)
        gradient_accumulation_steps=4,       # Effective batch size = 8 (2 * 4)
        optim="paged_adamw_8bit",            # Highly memory-efficient optimizer
        logging_steps=10,
        learning_rate=2e-4,                  # Standard learning rate for LoRA
        fp16=not supports_bf16,              # Fallback to fp16 if bf16 is not supported
        bf16=supports_bf16,                  # Use bf16 if supported
        max_grad_norm=0.3,
        warmup_steps=10,                     # Explicit warmup steps instead of deprecated warmup_ratio
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        report_to="none",                    # Set to "wandb" if you have a Weights & Biases account
        dataset_text_field="text",           # SFT-specific argument moved to SFTConfig
        max_length=512,                      # SFT-specific argument moved to SFTConfig
        packing=False,                       # SFT-specific argument moved to SFTConfig
    )

    # ----------------------------------------------------
    # SFT Trainer
    # ----------------------------------------------------
    print("\n--- 5. Starting Training ---")
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()

    print(f"\nTraining finished! Saving LoRA adapter to {adapter_output_dir}...")
    trainer.model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)

    # ----------------------------------------------------
    # Clear Memory
    # ----------------------------------------------------
    print("\n--- 6. Clearing Memory to Prepare for Merging ---")
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Merge LoRA Adapter with Base Model
    # ----------------------------------------------------
    print("\n--- 7. Loading Base Model to Merge Adapter ---")
    # Reload the base model on GPU to perform the merge (use bfloat16 for Gemma 2 if supported)
    merge_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(f"Loading base model in {merge_dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=merge_dtype,
        device_map="auto",  # Load on GPU/auto since training is finished and VRAM is empty
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_output_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_output_dir)

    print("Merging weights...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to {merged_output_dir}...")
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)

    print("\n==========================================")
    print("Fine-tuning and weight merging completed successfully!")
    print(f"Merged model is stored at: {merged_output_dir}")
    print("==========================================")

if __name__ == "__main__":
    main()

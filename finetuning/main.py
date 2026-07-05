import argparse
from src.utils import setup_logging, check_gpu_and_setup, clear_memory
from finetuning.configs.config import load_config
from src.model import load_tokenizer, load_base_model
from src.data import load_and_format_dataset
from src.trainer import get_lora_config, get_sft_config, run_training
from src.merger import merge_and_save_model
import logging

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Modular QLoRA Fine-tuning Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml", 
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    # 1. Load Config & Hardware Check
    config = load_config(args.config)
    supports_bf16 = check_gpu_and_setup()
    
    # 2. Data Preparation
    tokenizer = load_tokenizer(config)
    dataset = load_and_format_dataset(config, tokenizer)
    
    # 3. Model Preparation (4-bit)
    model = load_base_model(config, supports_bf16, quantized=True)
    
    # 4. Training Setup & Execution
    peft_config = get_lora_config(config)
    sft_config = get_sft_config(config, supports_bf16)
    adapter_output_dir = config['paths']['adapter_output_dir']
    
    run_training(model, tokenizer, dataset, peft_config, sft_config, adapter_output_dir)
    
    # 5. Cleanup & Merging
    del model
    clear_memory()
    
    merge_and_save_model(config, supports_bf16)
    
    logging.info("Entire Finetuning Pipeline completed successfully!")

if __name__ == "__main__":
    main()

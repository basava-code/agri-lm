import logging
from peft import PeftModel
from src.model import load_base_model, load_tokenizer

def merge_and_save_model(config: dict, supports_bf16: bool):
    paths_cfg = config['paths']
    adapter_dir = paths_cfg['adapter_output_dir']
    merged_dir = paths_cfg['merged_output_dir']
    
    logging.info("Loading Base Model to Merge Adapter...")
    base_model = load_base_model(config, supports_bf16, quantized=False)
    
    logging.info(f"Loading LoRA adapter from {adapter_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    logging.info("Merging weights...")
    merged_model = peft_model.merge_and_unload()
    
    logging.info(f"Saving merged model to {merged_dir}...")
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    
    tokenizer = load_tokenizer(config)
    tokenizer.save_pretrained(merged_dir)
    
    logging.info("Model merging and saving completed successfully!")

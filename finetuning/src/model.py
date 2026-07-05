import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def load_tokenizer(config: dict):
    model_id = config['model']['name']
    trust_remote = config['model'].get('trust_remote_code', True)
    
    logging.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_base_model(config: dict, supports_bf16: bool, quantized: bool = True):
    model_id = config['model']['name']
    trust_remote = config['model'].get('trust_remote_code', True)
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16

    if quantized:
        logging.info("Configuring 4-bit quantization (NF4 + Double Quant)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logging.info(f"Loading quantized base model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=trust_remote,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logging.info(f"Loading base model {model_id} in {compute_dtype} for merging...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=trust_remote,
        )
        
    return model

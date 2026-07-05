import gc
import logging
import torch

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def check_gpu_and_setup() -> bool:
    if not torch.cuda.is_available():
        logging.error("NO GPU DETECTED! Fine-tuning requires a GPU (T4 or higher).")
        raise RuntimeError("GPU is required for QLoRA fine-tuning.")
    
    supports_bf16 = torch.cuda.is_bf16_supported()
    logging.info(f"Hardware Detection: GPU Available = True | BF16 Supported = {supports_bf16}")
    return supports_bf16

def clear_memory():
    logging.info("Clearing GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()

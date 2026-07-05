import os
import logging
from datasets import load_dataset

def load_and_format_dataset(config: dict, tokenizer):
    dataset_path = config['dataset']['path']
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")

    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logging.info(f"Loaded {len(dataset)} examples.")

    def apply_chat_template(sample):
        formatted_text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        return {"text": formatted_text}

    logging.info("Applying chat template to dataset...")
    formatted_dataset = dataset.map(apply_chat_template, remove_columns=["messages"])
    
    logging.info(f"Example formatted text:\n{formatted_dataset[0]['text'][:300]}...")
    return formatted_dataset

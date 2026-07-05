import yaml
import logging

def load_config(config_path: str) -> dict:
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

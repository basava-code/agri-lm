"""LLM Factory Module.

Provides a unified factory to initialize LangChain chat models for different providers
(Ollama, and Huggingface) based on environment configuration.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure env variables are loaded
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "book_data_extractor" else current_dir
load_dotenv(project_root / ".env")

def get_llm(model_name: str | None = None, temperature: float = 0.0, **kwargs):
    """Initializes and returns a LangChain ChatModel based on environment configuration.
    
    Environment variables:
        LLM_PROVIDER: One of ollama, huggingface.
        MODEL_NAME: Name of the model to use.
        OLLAMA_BASE_URL: Base URL for Ollama (optional, defaults to http://localhost:11434).
    """
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower().strip()
    
    # Provider specific default models
    default_models = {
        "ollama": "llama3.1",
        "huggingface": "/workspace/Gemma API/gemma-4-E4B-it",
        "local_api": "Gemma4-e4B"
    }
    
    selected_model = model_name or os.getenv("MODEL_NAME") or default_models.get(provider)
    if not selected_model:
        raise ValueError(f"No model name provided and no default model configured for provider '{provider}'.")
        
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=selected_model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        
    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import pipeline
        import torch
        
        # Initialize the local HuggingFace pipeline for the GPU POD
        pipe = pipeline(
            "text-generation",
            model=selected_model,
            device_map="auto",
            max_new_tokens=4096,
            return_full_text=False,
            model_kwargs={"torch_dtype": torch.float16}
        )
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=hf_pipeline, **kwargs)
        
    elif provider == "local_api":
        from langchain_openai import ChatOpenAI
        # Connects to your running gemma_api.py FastAPI server without loading the model twice!
        base_url = os.getenv("LOCAL_API_BASE_URL", "http://localhost:8001/v1")
        return ChatOpenAI(
            model=selected_model,
            base_url=base_url,
            api_key="not-needed", # Local API does not require auth
            temperature=temperature,
            **kwargs
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported providers are: ollama, huggingface, local_api.")

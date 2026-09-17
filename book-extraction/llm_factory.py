"""LLM Factory Module.

Provides a unified factory to initialize LangChain chat models for different providers
(Ollama, HuggingFace, Google Gemini, Local API, OpenAI, NVIDIA) based on environment configuration.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure env variables are loaded
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name in ("book_data_extractor", "unit_test") else current_dir
load_dotenv(project_root / ".env")

def get_llm(model_name: str | None = None, temperature: float = 0.0, **kwargs):
    """Initializes and returns a LangChain ChatModel based on environment configuration.
    
    Environment variables:
        LLM_PROVIDER: One of huggingface, google, ollama, local_api, openai, nvidia.
        MODEL_NAME: Name or HuggingFace repo ID of the model to use.
        OLLAMA_BASE_URL: Base URL for Ollama (optional, defaults to http://localhost:11434).
        LOCAL_API_BASE_URL: Base URL for local vLLM/OpenAI server (optional, defaults to http://localhost:8001/v1).
    """
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower().strip()
    
    # Provider specific default models
    default_models = {
        "huggingface": "unsloth/gemma-4-E4B-it",
        "google": "gemini-2.5-flash",
        "ollama": "llama3.1",
        "local_api": "unsloth/gemma-4-E4B-it",
        "openai": "gpt-4o-mini",
        "nvidia": "meta/llama-3.1-70b-instruct"
    }
    
    selected_model = model_name or os.getenv("MODEL_NAME") or default_models.get(provider, "unsloth/gemma-4-E4B-it")
    if not selected_model:
        raise ValueError(f"No model name provided and no default model configured for provider '{provider}'.")
        
    # Auto-switch to Ollama if the model contains a colon (like 'gemma4:26b') and provider was left as default
    if provider == "huggingface" and ":" in selected_model and not os.getenv("LLM_PROVIDER"):
        print(f"[LLM Factory] Auto-detected Ollama model format ('{selected_model}'). Switching provider to 'ollama'.")
        provider = "ollama"
        
    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import pipeline
        import torch
        
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        
        pipe = pipeline(
            "text-generation",
            model=selected_model,
            device_map="auto",
            max_new_tokens=4096,
            return_full_text=False,
            model_kwargs={"torch_dtype": torch_dtype}
        )
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        
        # ChatHuggingFace expects an HF hub repo_id format (e.g., 'org/model').
        # If selected_model is a local path or custom name, handle exceptions gracefully.
        try:
            if "/" in selected_model and not os.path.exists(selected_model):
                return ChatHuggingFace(llm=hf_pipeline, model_id=selected_model, **kwargs)
            else:
                return ChatHuggingFace(llm=hf_pipeline, **kwargs)
        except Exception:
            return hf_pipeline

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
            temperature=temperature,
            **kwargs
        )
        
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=selected_model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        
    elif provider == "local_api":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("LOCAL_API_BASE_URL", "http://localhost:8001/v1")
        return ChatOpenAI(
            model=selected_model,
            base_url=base_url,
            api_key="not-needed",
            temperature=temperature,
            **kwargs
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            **kwargs
        )

    elif provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=selected_model,
            temperature=temperature,
            **kwargs
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Supported providers are: huggingface, google, ollama, local_api, openai, nvidia.")

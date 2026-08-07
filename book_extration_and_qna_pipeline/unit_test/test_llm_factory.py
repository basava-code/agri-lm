import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path to run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from book_data_extractor.llm_factory import get_llm
except ImportError:
    from llm_factory import get_llm

def test_factory():
    print("Testing LLM Factory initialization...")
    
    # 1. Test google initialization
    print("\n--- Testing Google Generative AI ---")
    try:
        os.environ["LLM_PROVIDER"] = "google"
        os.environ["MODEL_NAME"] = "gemini-2.5-flash"
        llm = get_llm()
        print(f"Successfully initialized: {type(llm).__name__} (Model: {llm.model})")
        
        # Test generation
        print("Invoking test prompt...")
        response = llm.invoke("State one key agricultural recommendation for growing rice in simple terms.")
        print(f"Response:\n{response.content.strip()}")
    except Exception as e:
        print(f"Expected failure for google provider: {e}")

    # 2. Test ollama initialization
    print("\n--- Testing Ollama (Local LLM) ---")
    try:
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["MODEL_NAME"] = "llama3.2:3b"
        llm = get_llm()
        print(f"Successfully initialized: {type(llm).__name__} (Model: {llm.model})")
        
        # Test generation
        print("Invoking test prompt...")
        response = llm.invoke("State one key agricultural recommendation for growing rice in simple terms.")
        print(f"Response:\n{response.content.strip()}")
    except Exception as e:
        print(f"Error testing ollama provider: {e}")

    # 3. Test nvidia initialization
    print("\n--- Testing NVIDIA (NVIDIA AI Endpoints) ---")
    try:
        os.environ["LLM_PROVIDER"] = "nvidia"
        if "MODEL_NAME" in os.environ:
            del os.environ["MODEL_NAME"]
        llm = get_llm()
        print(f"Successfully initialized: {type(llm).__name__} (Model: {llm.model})")
        
        # Test generation
        print("Invoking test prompt...")
        response = llm.invoke("State one key agricultural recommendation for growing rice in simple terms.")
        print(f"Response:\n{response.content.strip()}")
    except Exception as e:
        print(f"Error testing nvidia provider: {e}")

if __name__ == "__main__":
    test_factory()

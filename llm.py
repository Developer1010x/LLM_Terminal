# llm.py

import requests
import os

DEFAULT_LLM = "llama3.2:latest"
LLM_CONFIG_FILE = "llm_model.txt"

def get_model_name():
    """Reads the LLM model name from file or returns the default."""
    try:
        if os.path.exists(LLM_CONFIG_FILE):
            with open(LLM_CONFIG_FILE, "r") as f:
                model_name = f.read().strip()
                if model_name:
                    return model_name
    except Exception as e:
        print(f"Error reading LLM config: {e}")
    return DEFAULT_LLM

def ask_llm(prompt):
    """Sends a prompt to the selected LLM model."""
    model = get_model_name()
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error communicating with LLM: {e}"

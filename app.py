# ============================================================
# Alternative: Use Hugging Face Inference Client
# ============================================================

import streamlit as st
from huggingface_hub import InferenceClient
import time

HF_MODEL_NAME = "taibitfd/geocodegpt"
HF_TOKEN = "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"

@st.cache_resource
def get_client():
    """Get the inference client"""
    return InferenceClient(model=HF_MODEL_NAME, token=HF_TOKEN)

def generate_with_client(prompt, max_tokens=512, temperature=0.7):
    """Generate using the client"""
    try:
        client = get_client()
        
        system = """You are a Google Earth Engine expert. Generate only JavaScript code. 
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
        
        full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
        
        response = client.text_generation(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95
        )
        
        # Clean up response
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        elif "```javascript" in response:
            response = response.split("```javascript")[-1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
            
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"

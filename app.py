# ============================================================
# STREAMLIT APP FOR FREE HOSTING
# ============================================================

# app.py - Deploy to Streamlit Cloud for free

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess

# Download model from Drive on first run
@st.cache_resource
def load_model():
    MODEL_PATH = "/tmp/geocode_model"
    
    # Download if not exists
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive (first time only)...")
        os.makedirs(MODEL_PATH, exist_ok=True)
        # Use gdown to download from Drive
        # Replace with your actual Google Drive file/folder ID
        subprocess.run(["gdown", "--folder", "YOUR_FOLDER_ID", "-O", MODEL_PATH])
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

st.title("🌍 GeoCode-GPT")
st.write("Generate Google Earth Engine JavaScript code")

# Load model
tokenizer, model = load_model()

# Input
prompt = st.text_area("🌍 Describe what you want to do:", height=100)
max_tokens = st.slider("Max Tokens", 256, 2048, 1024)
temperature = st.slider("Temperature", 0.1, 1.0, 0.7)

if st.button("Generate Code"):
    with st.spinner("Generating..."):
        system = "You are a Google Earth Engine expert. Generate only JavaScript code. No markdown."
        full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        
        st.code(response, language="javascript")

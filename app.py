# ============================================================
# DEBUG VERSION - SEE THE PROCESS
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

HF_MODEL_NAME = "taibitfd/geocodegpt"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

st.title("🔍 Debug: GeoCode-GPT")

tokenizer, model = load_model()

if tokenizer and model:
    st.success("✅ Model loaded!")
    
    prompt = st.text_area("Enter your prompt:", "Calculate NDVI using Sentinel-2")
    
    if st.button("Generate"):
        with st.status("Generating...", expanded=True) as status:
            status.write("Step 1: Tokenizing input...")
            
            system = "You are a Google Earth Engine expert. Generate only JavaScript code."
            full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
            
            inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
            status.write("✅ Tokenized!")
            
            status.write("Step 2: Generating (this takes 10-30 seconds)...")
            start = time.time()
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id
                    )
                elapsed = time.time() - start
                status.write(f"✅ Generated in {elapsed:.1f}s!")
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "CODE:" in response:
                    response = response.split("CODE:")[-1].strip()
                
                status.update(label="✅ Done!", state="complete")
                st.code(response, language="javascript")
                
            except Exception as e:
                status.update(label="❌ Error!", state="error")
                st.error(f"Error: {str(e)}")

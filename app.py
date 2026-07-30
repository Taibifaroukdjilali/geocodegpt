# ============================================================
# STREAMLIT APP - REAL AI RESPONSES (CPU SHARDING)
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoCode-GPT - Real AI Edition")
st.caption("⚡ Using your 25GB model with sharding")

# ============================================================
# LOAD MODEL WITH SHARDING
# ============================================================

@st.cache_resource
def load_model_sharded():
    """Load model with CPU sharding for real AI responses"""
    try:
        with st.spinner("🧠 Loading model shards (this takes 3-5 minutes)..."):
            model_name = "taibitfd/geocodegpt"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with CPU sharding
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            st.success("✅ Model loaded! Ready for real AI responses.")
            return tokenizer, model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================
# REAL AI GENERATION FUNCTION
# ============================================================

def generate_real_ai(prompt, max_tokens=512, temperature=0.7):
    """Generate REAL AI responses using your model"""
    try:
        system = """You are a Google Earth Engine expert. Generate only JavaScript code.
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
        
        full_prompt = f"{system}\n\nUser request: {prompt}\n\nCODE:"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Generate with real AI
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        elif "```javascript" in response:
            response = response.split("```javascript")[-1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        
        # Clean memory
        gc.collect()
        
        return response
        
    except Exception as e:
        gc.collect()
        return f"❌ AI Error: {str(e)}"

# ============================================================
# LOAD THE MODEL
# ============================================================

tokenizer, model = load_model_sharded()

if tokenizer is None or model is None:
    st.error("❌ Model failed to load. Please refresh and try again.")
    st.stop()

# ============================================================
# STREAMLIT UI
# ============================================================

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 128, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    st.subheader("📊 Status")
    st.success("✅ AI Model Ready")
    st.caption("🧠 Real AI responses")
    st.caption("💻 Running on CPU")
    st.caption("⏱️ Generation: 10-30 seconds")

# Main
st.subheader("🌍 Describe what you want to do:")
prompt = st.text_area(
    "",
    height=100,
    placeholder="Example: Calculate NDVI using Sentinel-2 for California"
)

# Quick examples
cols = st.columns(4)
with cols[0]:
    if st.button("🛰️ NDVI", use_container_width=True):
        prompt = "Calculate NDVI using Sentinel-2 for California"
        st.rerun()
with cols[1]:
    if st.button("🌿 True Color", use_container_width=True):
        prompt = "Create a true color composite of Sentinel-2"
        st.rerun()
with cols[2]:
    if st.button("💧 Water", use_container_width=True):
        prompt = "Create NDWI water body detection using Sentinel-2"
        st.rerun()
with cols[3]:
    if st.button("📤 Export", use_container_width=True):
        prompt = "Export Landsat 8 image to Google Drive"
        st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("🤖 AI is thinking (10-30 seconds)..."):
            response = generate_real_ai(prompt, max_tokens, temperature)
            
            if response.startswith("❌"):
                st.error(response)
            else:
                st.subheader("💻 Generated Code (Real AI)")
                st.code(response, language="javascript")
                
                st.download_button(
                    label="📥 Download",
                    data=response,
                    file_name="code.js",
                    mime="text/javascript"
                )

st.divider()
st.caption("📌 Generated code for Google Earth Engine JavaScript API")

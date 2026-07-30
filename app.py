# ============================================================
# app.py - MEMORY OPTIMIZED FOR STREAMLIT CLOUD
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_NAME = "taibitfd/geocodegpt"

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# MODEL LOADER WITH MEMORY OPTIMIZATIONS
# ============================================================

@st.cache_resource
def load_model():
    """Load model with memory optimizations for Streamlit Cloud"""
    try:
        st.info("📥 Loading model from Hugging Face...")
        
        # Step 1: Load tokenizer
        st.info("Step 1/3: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Step 2: Load model with memory optimizations
        st.info("Step 2/3: Loading model weights (this takes 2-3 minutes)...")
        
        # Use CPU mode with memory optimizations for Streamlit Cloud
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float32,  # Use float32 for CPU stability
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        # Try to move to GPU if available
        if torch.cuda.is_available():
            st.info("✅ GPU available, moving model to GPU...")
            model = model.to('cuda')
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Troubleshooting:**
        1. The model might be too large for the free tier
        2. Try the 4-bit quantization version below
        3. Consider upgrading to a paid tier
        """)
        return None, None

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_code(tokenizer, model, prompt, max_tokens, temperature):
    """Generate Earth Engine JavaScript code"""
    try:
        system = "You are a Google Earth Engine expert. Generate only JavaScript code. No markdown."
        full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate with fewer tokens for memory
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),  # Limit for memory
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        
        # Clean memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"❌ Error: {str(e)}"

# ============================================================
# UI
# ============================================================

st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("Generate Google Earth Engine JavaScript code using AI.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 128, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    
    # Memory info
    st.caption("💾 Memory Mode: Optimized")

# Load model with progress
with st.spinner("🔄 Loading model (this may take 3-5 minutes)..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Failed to load model.")
    st.info("""
    **Solutions:**
    1. Wait a few minutes and refresh
    2. Check the model: https://huggingface.co/taibitfd/geocodegpt
    3. Try the alternative version below
    """)
    
    # Alternative: Use original model
    if st.button("🔄 Try alternative model (lzq677/GeoCode-GPT)"):
        st.cache_resource.clear()
        st.rerun()
    
    st.stop()

# Main interface
prompt = st.text_area(
    "🌍 Describe what you want to do:",
    height=120,
    placeholder="Example: Create a true color composite of Sentinel-2 for California"
)

# Quick examples
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🛰️ True Color", use_container_width=True):
        prompt = "Create a true color composite of Sentinel-2 for California"
        st.rerun()
with col2:
    if st.button("🌿 NDVI", use_container_width=True):
        prompt = "Calculate NDVI using Landsat 8 for Brazil"
        st.rerun()
with col3:
    if st.button("💧 Water", use_container_width=True):
        prompt = "Create an NDWI water body detection using Sentinel-2"
        st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("🤖 Generating code..."):
            response = generate_code(tokenizer, model, prompt, max_tokens, temperature)
            
            if response.startswith("❌"):
                st.error(response)
            else:
                st.subheader("💻 Generated Code")
                st.code(response, language="javascript")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download",
                        data=response,
                        file_name="earth_engine_code.js",
                        mime="text/javascript"
                    )
                with col2:
                    st.button("📋 Copy", on_click=lambda: st.write("Select code and copy"))

# Footer
st.divider()
st.caption("📌 Generated code for Google Earth Engine JavaScript API")

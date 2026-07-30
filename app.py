# ============================================================
# STREAMLIT APP - LOAD ONLY WHAT'S NEEDED (SHARDING)
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ============================================================
# MEMORY OPTIMIZATION SETTINGS
# ============================================================

# Force memory-efficient loading
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(
    page_title="🌍 GeoCode-GPT - Lightweight",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoCode-GPT - Lightweight Edition")
st.caption("⚡ Using model sharding - loads only what's needed!")

# ============================================================
# MODEL LOADER WITH SHARDING
# ============================================================

@st.cache_resource
def load_model_sharded():
    """Load model using sharding - only loads what fits in memory"""
    
    try:
        with st.spinner("🧠 Loading model shards (only ~6GB!)..."):
            
            # Step 1: Load tokenizer (tiny)
            tokenizer = AutoTokenizer.from_pretrained("taibitfd/geocodegpt")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Step 2: Load model with sharding
            model = AutoModelForCausalLM.from_pretrained(
                "taibitfd/geocodegpt",
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                # KEY: Only load shards as needed
                max_memory={
                    0: "6GB",      # GPU 0: Only 6GB
                    "cpu": "8GB"   # CPU: 8GB buffer
                }
            )
            
            # Step 3: Show memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                st.success(f"✅ Model loaded with {memory_used:.1f}GB GPU memory (saved ~19GB!)")
            else:
                st.success("✅ Model loaded on CPU with sharding")
            
            return tokenizer, model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================
# LOAD THE MODEL
# ============================================================

tokenizer, model = load_model_sharded()

if tokenizer is None or model is None:
    st.warning("⚠️ Model failed to load. Trying fallback...")
    st.stop()

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_code(prompt, max_tokens=512, temperature=0.7):
    """Generate Earth Engine code with memory cleanup"""
    try:
        system = """You are a Google Earth Engine expert. Generate only JavaScript code.
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
        
        full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        elif "```javascript" in response:
            response = response.split("```javascript")[-1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        
        # Clean memory after generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
        
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error: {str(e)}"

# ============================================================
# STREAMLIT UI
# ============================================================

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider(
        "Max Tokens",
        min_value=128,
        max_value=1024,
        value=512,
        step=64,
        help="Maximum length of generated code"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher = more creative, Lower = more deterministic"
    )
    
    st.divider()
    st.subheader("📊 Model Status")
    
    if torch.cuda.is_available():
        st.caption(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.caption(f"💾 Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")
    else:
        st.caption("🖥️ Using CPU mode")
    
    st.caption("⚡ Sharded loading (only ~6GB)")

# Main input
prompt = st.text_area(
    "🌍 Describe what you want to do:",
    height=120,
    placeholder="Example: Calculate NDVI using Sentinel-2 for California",
    help="Be specific about the region, dataset, and analysis"
)

# Quick example buttons
st.subheader("💡 Quick Examples")
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

# Generate button
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Generating code...")
        
        with st.spinner("🤖 Thinking..."):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if start_time:
                start_time.record()
            
            response = generate_code(prompt, max_tokens, temperature)
            
            if start_time:
                end_time = torch.cuda.Event(enable_timing=True)
                end_time.record()
                torch.cuda.synchronize()
                elapsed = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            else:
                elapsed = "unknown"
            
            status_placeholder.empty()
            
            if response.startswith("Error"):
                st.error(response)
            else:
                st.success(f"✅ Code generated in {elapsed}s" if elapsed != "unknown" else "✅ Code generated!")
                st.subheader("💻 Generated Code")
                st.code(response, language="javascript")
                
                # Download button
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Code",
                        data=response,
                        file_name="earth_engine_code.js",
                        mime="text/javascript",
                        use_container_width=True
                    )
                with col2:
                    if st.button("📋 Copy", use_container_width=True):
                        st.write("Select code and press Ctrl+C to copy")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")
st.caption("⚡ Lightweight mode: only loads shards as needed (~6GB memory)")

# ============================================================
# MEMORY CLEANUP ON APP CLOSE
# ============================================================

import atexit
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

atexit.register(cleanup)

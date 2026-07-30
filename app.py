# ============================================================
# app.py - GeoCode-GPT with Google Drive Integration
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess
import shutil
import json
import time
import sys

# ============================================================
# CONFIGURATION - YOUR CORRECT FOLDER ID
# ============================================================

DRIVE_FOLDER_ID = "1738_TVggvUcf-5e8Bxz7KJ-_281kW6vW"
MODEL_PATH = "/tmp/geocode_model"

# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def check_and_install_dependencies():
    """Check and install missing dependencies"""
    try:
        import sentencepiece
        return True
    except ImportError:
        st.warning("⚠️ sentencepiece not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece"], 
                      capture_output=True)
        try:
            import sentencepiece
            st.success("✅ sentencepiece installed successfully!")
            return True
        except:
            st.error("❌ Failed to install sentencepiece")
            return False

def download_model_from_drive():
    """Download model from Google Drive using gdown"""
    try:
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        st.info("📥 Downloading model from Google Drive...")
        st.info("⏳ This may take 5-10 minutes...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Downloading model files...")
        progress_bar.progress(30)
        
        # Download the folder
        cmd = f"gdown --folder {DRIVE_FOLDER_ID} -O {MODEL_PATH}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        progress_bar.progress(80)
        status_text.text("Processing downloaded files...")
        
        if result.returncode == 0:
            files = os.listdir(MODEL_PATH)
            model_files = [f for f in files if f.endswith(('.bin', '.safetensors'))]
            
            if model_files:
                progress_bar.progress(100)
                status_text.text("✅ Download complete!")
                st.success(f"✅ Downloaded {len(model_files)} model weight files!")
                return True
            else:
                st.warning("⚠️ Downloaded folder but no model weights found.")
                return False
        else:
            st.error(f"❌ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        st.error(f"❌ Download error: {e}")
        return False

def download_with_retry():
    """Download with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        st.info(f"📥 Download attempt {attempt + 1}/{max_retries}...")
        if download_model_from_drive():
            return True
        st.warning(f"Attempt {attempt + 1} failed. Retrying...")
        time.sleep(5)
    return False

# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():
    """Load the model with caching for better performance"""
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        st.error("❌ Missing required dependencies")
        return None, None
    
    # Check if model exists and is valid
    if os.path.exists(MODEL_PATH):
        files = os.listdir(MODEL_PATH)
        model_files = [f for f in files if f.endswith(('.bin', '.safetensors'))]
        
        # Check for tokenizer files
        has_tokenizer = any(f in files for f in ['tokenizer.json', 'tokenizer.model'])
        
        if model_files and has_tokenizer:
            st.info(f"✅ Found {len(model_files)} model files locally. Loading...")
        else:
            st.warning("📦 Model files found but incomplete. Re-downloading...")
            shutil.rmtree(MODEL_PATH)
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            if not download_with_retry():
                st.error("❌ Failed to download model. Please check your Drive link.")
                return None, None
    else:
        # First time - download from Drive
        st.info("📦 Model not found locally. Downloading from Google Drive...")
        if not download_with_retry():
            st.error("❌ Failed to download model. Please check your Drive link.")
            return None, None
    
    # Load the model
    try:
        st.info("🧠 Loading model into memory...")
        
        # Load tokenizer with use_fast=False to use slow tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=False  # Use slow tokenizer with sentencepiece
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # Show detailed troubleshooting
        st.info("""
        **Troubleshooting:**
        1. Make sure `sentencepiece` is installed in requirements.txt
        2. Your Drive folder should contain:
           - config.json ✓
           - model-*.safetensors files ✓
           - tokenizer.json or tokenizer.model (missing!)
        3. Check that folder permissions are set to "Anyone with the link"
        """)
        
        # Show what files are in the directory
        if os.path.exists(MODEL_PATH):
            files = os.listdir(MODEL_PATH)
            st.write("📁 Files in model directory:", files)
            
            # Check specifically for tokenizer files
            tokenizer_files = [f for f in files if 'tokenizer' in f]
            if tokenizer_files:
                st.write("✅ Found tokenizer files:", tokenizer_files)
            else:
                st.error("❌ No tokenizer files found! This is the problem.")
                st.info("""
                **Solution:** Your Drive folder is missing tokenizer files.
                You need to upload:
                - tokenizer.json
                - tokenizer.model
                - tokenizer_config.json
                - special_tokens_map.json
                
                These files should be in your model folder. Please upload them to Drive.
                """)
        
        return None, None

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_code(tokenizer, model, prompt, max_tokens, temperature):
    """Generate Earth Engine JavaScript code from prompt"""
    system = "You are a Google Earth Engine expert. Generate only JavaScript code. No markdown."
    full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "CODE:" in response:
        response = response.split("CODE:")[-1].strip()
    
    return response

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("""
Generate Google Earth Engine JavaScript code using AI.
Describe what you want to do, and the model will generate the code!
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 256, 2048, 1024)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    
    # Show model status
    st.subheader("📊 Model Status")
    if os.path.exists(MODEL_PATH):
        files = os.listdir(MODEL_PATH)
        model_files = [f for f in files if f.endswith(('.bin', '.safetensors'))]
        tokenizer_files = [f for f in files if 'tokenizer' in f]
        
        if model_files and tokenizer_files:
            st.success(f"✅ Model ready")
            st.caption(f"   - {len(model_files)} weight files")
            st.caption(f"   - {len(tokenizer_files)} tokenizer files")
        else:
            st.warning("⏳ Model not fully loaded")
            if not model_files:
                st.caption("⚠️ Missing model weights")
            if not tokenizer_files:
                st.caption("⚠️ Missing tokenizer files")
    else:
        st.info("⏳ First time setup...")
    
    st.divider()
    st.caption("📁 Model from Google Drive")

# Load model
with st.spinner("🔄 Loading model (first time may take 5-10 minutes)..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Failed to load model. Please check the error messages above.")
    st.stop()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "🌍 Describe what you want to do:",
        height=150,
        placeholder="Example: Create a true color composite of Sentinel-2 for California"
    )

with col2:
    st.subheader("💡 Example Prompts")
    examples = [
        "Create a true color composite of Sentinel-2 for California",
        "Calculate NDVI for a region in Brazil using Landsat 8",
        "Export a Landsat 8 image to Google Drive",
        "Create an NDWI water body detection using Sentinel-2",
        "Generate a land cover classification using Random Forest"
    ]
    
    for example in examples:
        if st.button(example[:30] + "..." if len(example) > 30 else example, 
                     key=example, use_container_width=True):
            prompt = example
            st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("🤖 Generating code..."):
            try:
                response = generate_code(tokenizer, model, prompt, max_tokens, temperature)
                
                st.subheader("💻 Generated Code")
                st.code(response, language="javascript")
                
                st.download_button(
                    label="📥 Download Code",
                    data=response,
                    file_name="earth_engine_code.js",
                    mime="text/javascript",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error generating code: {e}")

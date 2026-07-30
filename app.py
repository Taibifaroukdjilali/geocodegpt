# ============================================================
# app.py - Enhanced with Google Drive Download
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess
import shutil
from pathlib import Path

# Model configuration
MODEL_PATH = "/tmp/geocode_model"
DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE"  # Replace with your actual folder ID

# Function to download model from Drive
def download_model_from_drive():
    """Download model from Google Drive using gdown"""
    try:
        # Install gdown if not available
        subprocess.run(["pip", "install", "gdown"], capture_output=True)
        
        # Create directory
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # Download the entire folder
        st.info("📥 Downloading model from Google Drive (first time only)...")
        st.info("⏳ This may take 5-10 minutes...")
        
        # Method 1: Download entire folder
        # Replace with your actual Google Drive folder ID
        cmd = f"gdown --folder {DRIVE_FOLDER_ID} -O {MODEL_PATH} --remaining-ok"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Download failed: {result.stderr}")
            return False
            
        # Check if model files exist
        files = os.listdir(MODEL_PATH)
        if any(f.endswith(('.bin', '.safetensors')) for f in files):
            st.success("✅ Model downloaded successfully!")
            return True
        else:
            st.warning("No model weights found. Trying alternative download method...")
            return download_model_alternative()
            
    except Exception as e:
        st.error(f"❌ Download error: {e}")
        return False

def download_model_alternative():
    """Alternative download method using direct file link"""
    try:
        # Replace with your direct file link from Google Drive
        # Format: https://drive.google.com/uc?id=FILE_ID
        file_id = "YOUR_FILE_ID_HERE"  # Replace with actual file ID
        
        cmd = f"gdown https://drive.google.com/uc?id={file_id} -O {MODEL_PATH}/pytorch_model.bin"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("✅ Model downloaded successfully!")
            return True
        else:
            st.error(f"Alternative download failed: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Alternative download error: {e}")
        return False

# Cache the model loading for better performance
@st.cache_resource
def load_model():
    """Load the model with caching"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not any(os.listdir(MODEL_PATH)):
        st.info("📦 Model not found locally. Downloading from Google Drive...")
        if not download_model_from_drive():
            st.error("Failed to download model. Please check your Drive links.")
            return None, None
    
    try:
        st.info("📥 Loading model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations for Streamlit Cloud
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
        st.info("If you're seeing this, the model files might be incomplete.")
        st.info("Check your Google Drive folder and try again.")
        return None, None

# Streamlit UI
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
    max_tokens = st.slider("Max Tokens", 256, 2048, 1024, 
                          help="Maximum length of generated code")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05,
                           help="Higher = more creative, Lower = more deterministic")
    
    st.divider()
    st.caption("📁 Model stored in Google Drive")
    st.caption(f"Path: `{MODEL_PATH}`")
    
    # Show model status
    if os.path.exists(MODEL_PATH):
        files = os.listdir(MODEL_PATH)
        st.success(f"✅ Model loaded ({len(files)} files)")
    else:
        st.warning("⏳ Model not loaded")

# Load model
with st.spinner("Loading model (this may take a few minutes on first run)..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Failed to load model. Please check your Google Drive links and try again.")
    st.stop()

# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "🌍 Describe what you want to do:",
        height=150,
        placeholder="Example: Create a true color composite of Sentinel-2 for California",
        help="Be specific about the region, dataset, and analysis"
    )
    
    examples = [
        "Create a true color composite of Sentinel-2 for California",
        "Calculate NDVI for a region in Brazil using Landsat 8",
        "Export a Landsat 8 image to Google Drive",
        "Make a time-lapse animation of forest cover change",
        "Create an NDWI water body detection using Sentinel-2",
        "Generate a land cover classification using Random Forest"
    ]
    
    example_prompt = st.selectbox("💡 Example prompts:", [""] + examples)
    if example_prompt:
        prompt = example_prompt
        st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description of what you want to do.")
    else:
        with st.spinner("🤖 Generating code..."):
            try:
                # Prepare prompt
                system = "You are a Google Earth Engine expert. Generate only JavaScript code. No markdown."
                full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
                
                # Tokenize
                inputs = tokenizer(
                    full_prompt, 
                    return_tensors="pt", 
                    max_length=2048, 
                    truncation=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "CODE:" in response:
                    response = response.split("CODE:")[-1].strip()
                
                # Display result
                st.subheader("💻 Generated Code")
                st.code(response, language="javascript")
                
                # Add download button
                st.download_button(
                    label="📥 Download Code",
                    data=response,
                    file_name="earth_engine_code.js",
                    mime="text/javascript"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating code: {e}")
                st.info("Try adjusting the temperature or max tokens settings.")

# Footer
st.divider()
st.caption("🔧 Built with 🤗 Transformers and Streamlit")
st.caption(f"📊 Model loaded from: Google Drive ({MODEL_PATH})")

# ============================================================
# app.py - GeoCode-GPT with Multiple Upload Methods
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import zipfile
import shutil
import json
import glob

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/tmp/geocode_model"
ZIP_PATH = "/tmp/model.zip"

# ============================================================
# FILE CHECKING FUNCTIONS
# ============================================================

def check_model_files():
    """Check what files are in the model directory"""
    if not os.path.exists(MODEL_PATH):
        return {"exists": False, "files": [], "weights": [], "tokenizer": []}
    
    files = os.listdir(MODEL_PATH)
    
    # Check for weight files
    weights = [f for f in files if f.endswith(('.bin', '.safetensors'))]
    
    # Check for tokenizer files
    tokenizer = [f for f in files if 'tokenizer' in f.lower()]
    
    # Check for config
    config = [f for f in files if f == 'config.json']
    
    return {
        "exists": True,
        "files": files,
        "weights": weights,
        "tokenizer": tokenizer,
        "config": config,
        "count": len(files)
    }

def check_zip_file():
    """Check if ZIP file exists and what's in it"""
    if not os.path.exists(ZIP_PATH):
        return None
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            return zip_ref.namelist()
    except:
        return None

# ============================================================
# MODEL LOADER WITH MULTIPLE METHODS
# ============================================================

@st.cache_resource
def load_model():
    """Load the model with multiple methods"""
    
    # First, check what we have
    file_status = check_model_files()
    zip_contents = check_zip_file()
    
    # Method 1: Check if model files exist directly
    if file_status["exists"] and file_status["weights"]:
        st.success(f"✅ Found {len(file_status['weights'])} model weight files")
        st.success(f"✅ Found {len(file_status['tokenizer'])} tokenizer files")
        return load_from_directory(MODEL_PATH)
    
    # Method 2: Try to extract from ZIP
    elif zip_contents:
        st.info(f"📦 Found ZIP file with {len(zip_contents)} files. Extracting...")
        try:
            # Remove existing directory
            if os.path.exists(MODEL_PATH):
                shutil.rmtree(MODEL_PATH)
            os.makedirs(MODEL_PATH, exist_ok=True)
            
            # Extract ZIP
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(MODEL_PATH)
            
            # Check extraction
            file_status = check_model_files()
            if file_status["weights"]:
                st.success("✅ Model extracted successfully!")
                return load_from_directory(MODEL_PATH)
            else:
                st.warning("⚠️ ZIP extracted but no model weights found")
        except Exception as e:
            st.error(f"❌ Failed to extract ZIP: {e}")
    
    # Method 3: Check for model files in subdirectories
    if file_status["exists"] and file_status["files"]:
        st.info("🔍 Searching for model files in subdirectories...")
        for root, dirs, files in os.walk(MODEL_PATH):
            for file in files:
                if file.endswith(('.bin', '.safetensors')):
                    # Move files to main directory
                    src = os.path.join(root, file)
                    dst = os.path.join(MODEL_PATH, file)
                    shutil.move(src, dst)
                    st.info(f"Moved {file} to main directory")
        
        # Check again
        file_status = check_model_files()
        if file_status["weights"]:
            st.success("✅ Found and organized model files!")
            return load_from_directory(MODEL_PATH)
    
    # No model found
    st.error("❌ No model files found!")
    show_upload_instructions()
    return None, None

def load_from_directory(path):
    """Load model from directory"""
    try:
        st.info("🧠 Loading model into memory...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # Show detailed troubleshooting
        file_status = check_model_files()
        st.info(f"📁 Files in directory: {file_status['files']}")
        
        if not file_status["config"]:
            st.error("❌ Missing config.json")
        if not file_status["weights"]:
            st.error("❌ Missing model weights (.bin or .safetensors)")
        if not file_status["tokenizer"]:
            st.warning("⚠️ Missing tokenizer files")
        
        return None, None

def show_upload_instructions():
    """Show instructions for uploading files"""
    st.info("""
    **📤 How to upload model files:**
    
    **Option A: Upload Individual Files**
    1. Go to your Streamlit Cloud app dashboard
    2. Click on the **"Files"** section in the left sidebar
    3. Create the directory `/tmp/geocode_model/`
    4. Upload all model files:
       - `config.json`
       - `model-00001-of-00006.safetensors` (all 6 parts)
       - `tokenizer.json`
       - `tokenizer.model`
       - `tokenizer_config.json`
       - `special_tokens_map.json`
    5. Refresh the app
    
    **Option B: Upload a ZIP File (Easier)**
    1. Create a ZIP file containing ALL model files
    2. Name it `model.zip`
    3. Upload it to `/tmp/model.zip` in your Streamlit Cloud app
    4. Refresh the app - it will auto-extract
    
    **Option C: Use Hugging Face (Recommended for Production)**
    1. Upload your model to Hugging Face Hub
    2. Update the code to load directly from Hugging Face
    """)

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
    file_status = check_model_files()
    
    if file_status["exists"]:
        if file_status["weights"]:
            st.success(f"✅ {len(file_status['weights'])} weight files")
            if file_status["tokenizer"]:
                st.success(f"✅ {len(file_status['tokenizer'])} tokenizer files")
            else:
                st.warning("⚠️ Missing tokenizer files")
            # Show file sizes
            total_size = 0
            for f in file_status["weights"][:3]:
                try:
                    size = os.path.getsize(os.path.join(MODEL_PATH, f)) / (1024**3)
                    total_size += size
                    st.caption(f"   - {f} ({size:.2f} GB)")
                except:
                    pass
            if len(file_status["weights"]) > 3:
                st.caption(f"   ... and {len(file_status['weights']) - 3} more")
            if total_size > 0:
                st.caption(f"📊 Total: ~{total_size:.1f} GB")
        else:
            st.warning(f"⚠️ {file_status['count']} files found, but no weights")
            st.caption(f"Files: {', '.join(file_status['files'][:5])}")
    else:
        st.warning("⏳ No model loaded")
        if os.path.exists(ZIP_PATH):
            st.info("📦 ZIP file found - will extract on load")
    
    st.divider()
    st.caption("🔧 Built with ❤️ using Transformers & Streamlit")

# Load model
with st.spinner("🔄 Loading model..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    show_upload_instructions()
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
        "Generate a land cover classification using Random Forest",
        "Create a time series chart of vegetation health"
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

st.divider()
st.caption("📌 Tip: Generated code is for Google Earth Engine JavaScript API")

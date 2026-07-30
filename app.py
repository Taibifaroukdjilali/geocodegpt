# ============================================================
# app.py - GeoCode-GPT with Direct File Upload
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import zipfile
import shutil
import json

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/tmp/geocode_model"
ZIP_PATH = "/tmp/model.zip"  # Optional: if you upload a zip file

# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():
    """Load the model from /tmp directory"""
    
    # Check if model files exist directly
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        st.info(f"📁 Found {len(os.listdir(MODEL_PATH))} files in model directory")
        files = os.listdir(MODEL_PATH)
        
        # Check for model weights
        model_files = [f for f in files if f.endswith(('.bin', '.safetensors'))]
        if model_files:
            st.success(f"✅ Found {len(model_files)} model weight files")
        else:
            st.error("❌ No model weight files found (.bin or .safetensors)")
            st.info("Please upload your model files to /tmp/geocode_model/")
            return None, None
        
        # Check for tokenizer files
        tokenizer_files = [f for f in files if 'tokenizer' in f]
        if tokenizer_files:
            st.success(f"✅ Found {len(tokenizer_files)} tokenizer files")
        else:
            st.warning("⚠️ Tokenizer files may be missing")
    
    # If model directory doesn't exist or is empty, try to extract from zip
    elif os.path.exists(ZIP_PATH):
        st.info("📦 Extracting model from uploaded ZIP file...")
        try:
            # Remove existing directory if it exists
            if os.path.exists(MODEL_PATH):
                shutil.rmtree(MODEL_PATH)
            
            # Extract zip
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(MODEL_PATH)
            st.success("✅ Model extracted successfully!")
            
            # Verify extraction
            files = os.listdir(MODEL_PATH)
            st.info(f"📁 Extracted {len(files)} files")
            
        except Exception as e:
            st.error(f"❌ Failed to extract model: {e}")
            st.info("Please upload your model files directly to /tmp/geocode_model/")
            return None, None
    else:
        # No model found anywhere
        st.error("❌ No model files found!")
        st.info("""
        **Please upload your model files:**
        1. In the Streamlit Cloud dashboard, go to the Files section
        2. Navigate to `/tmp/geocode_model/`
        3. Upload all model files (config.json, model-*.safetensors, tokenizer files)
        **OR**
        1. Upload a ZIP file named `model.zip` to `/tmp/`
        """)
        return None, None
    
    # Load the model
    try:
        st.info("🧠 Loading model into memory...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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
        1. Make sure your model directory contains:
           - config.json
           - model-*.safetensors files (all 6 parts)
           - tokenizer.json
           - tokenizer.model
           - tokenizer_config.json
           - special_tokens_map.json
        2. Check that the files are not corrupted
        3. Ensure all files are in `/tmp/geocode_model/`
        """)
        
        # Show what files are in the directory
        if os.path.exists(MODEL_PATH):
            files = os.listdir(MODEL_PATH)
            st.write("📁 Files in model directory:", files)
            
            # Check specifically for required files
            required_files = ['config.json']
            missing = [f for f in required_files if f not in files]
            if missing:
                st.error(f"❌ Missing required files: {missing}")
        
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

# Page configuration
st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# Header
st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("""
Generate Google Earth Engine JavaScript code using AI.
Describe what you want to do, and the model will generate the code!
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=2048,
        value=1024,
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
            
            # Show total size
            total_size = 0
            for f in model_files:
                try:
                    size = os.path.getsize(os.path.join(MODEL_PATH, f)) / (1024**3)
                    total_size += size
                except:
                    pass
            if total_size > 0:
                st.caption(f"📊 Total size: ~{total_size:.1f} GB")
        elif model_files:
            st.warning("⚠️ Model weights found, but tokenizer files missing")
        elif tokenizer_files:
            st.warning("⚠️ Tokenizer files found, but model weights missing")
        else:
            st.warning("⏳ Model files not found")
    else:
        st.info("⏳ No model loaded yet")
    
    st.divider()
    st.caption("🔧 Built with ❤️ using Transformers & Streamlit")
    st.caption("📁 Model loaded from /tmp/geocode_model/")

# Load model
with st.spinner("🔄 Loading model (this may take 2-3 minutes)..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Failed to load model. Please upload your model files.")
    st.info("""
    **How to upload model files:**
    1. Go to your Streamlit Cloud app dashboard
    2. Click on the "Files" section
    3. Navigate to `/tmp/geocode_model/`
    4. Upload all model files:
       - config.json
       - model-00001-of-00006.safetensors through model-00006-of-00006.safetensors
       - tokenizer.json
       - tokenizer.model
       - tokenizer_config.json
       - special_tokens_map.json
    5. Refresh the app
    """)
    st.stop()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "🌍 Describe what you want to do:",
        height=150,
        placeholder="Example: Create a true color composite of Sentinel-2 for California",
        help="Be specific about the region, dataset, and analysis"
    )

with col2:
    st.subheader("💡 Example Prompts")
    examples = [
        "Create a true color composite of Sentinel-2 for California",
        "Calculate NDVI for a region in Brazil using Landsat 8",
        "Export a Landsat 8 image to Google Drive",
        "Create an NDWI water body detection using Sentinel-2",
        "Generate a land cover classification using Random Forest",
        "Create a time series chart of vegetation health",
        "Detect deforestation using Sentinel-2 time series"
    ]
    
    for example in examples:
        if st.button(example[:30] + "..." if len(example) > 30 else example, 
                     key=example, use_container_width=True):
            prompt = example
            st.rerun()

# Generate button
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description of what you want to do.")
    else:
        with st.spinner("🤖 Generating code..."):
            try:
                # Generate code
                response = generate_code(tokenizer, model, prompt, max_tokens, temperature)
                
                # Display results
                st.subheader("💻 Generated Code")
                st.code(response, language="javascript")
                
                # Download button
                st.download_button(
                    label="📥 Download Code",
                    data=response,
                    file_name="earth_engine_code.js",
                    mime="text/javascript",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error generating code: {e}")
                st.info("Try adjusting the temperature or max tokens settings.")

# Footer
st.divider()
st.caption("📌 Tip: Generated code is for Google Earth Engine JavaScript API")

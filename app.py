# ============================================================
# app.py - GeoCode-GPT for Streamlit Cloud
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_NAME = "taibitfd/geocodegpt"

# Set environment variables for better performance
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# MODEL LOADER WITH PROPER CACHING
# ============================================================

@st.cache_resource
def load_model():
    """Load model from Hugging Face with caching and progress tracking"""
    try:
        # Create a status container for loading progress
        status_container = st.status("📥 Loading model from Hugging Face...", expanded=True)
        
        status_container.write("Step 1/3: Loading tokenizer...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        status_container.write("Step 2/3: Loading model weights (this takes 2-3 minutes)...")
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        status_container.write("Step 3/3: Model loaded successfully! ✅")
        status_container.update(label="✅ Model loaded successfully!", state="complete")
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Troubleshooting:**
        - Check if the model exists: https://huggingface.co/taibitfd/geocodegpt
        - Make sure the model is public
        - Refresh the page and try again
        """)
        return None, None

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_code(tokenizer, model, prompt, max_tokens, temperature):
    """Generate Earth Engine JavaScript code from prompt"""
    try:
        system = "You are a Google Earth Engine expert. Generate only JavaScript code. No markdown."
        full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
        
        # Tokenize input
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=2048, truncation=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
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
        
    except Exception as e:
        return f"❌ Error generating code: {str(e)}"

# ============================================================
# STREAMLIT UI
# ============================================================

# Page configuration
st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Model information
    st.subheader("📊 Model Info")
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("📁 Hosted on Hugging Face Hub")
    st.markdown(f"[🔗 View on Hugging Face](https://huggingface.co/{HF_MODEL_NAME})")
    
    # Check if model is loaded
    if 'model' in st.session_state:
        st.success("✅ Model ready")
    else:
        st.info("⏳ Loading model...")
    
    st.divider()
    st.caption("🔧 Built with ❤️ using Transformers & Streamlit")

# ============================================================
# MAIN CONTENT
# ============================================================

# Load model with a spinner
with st.spinner("🔄 Loading model from Hugging Face (this may take 2-3 minutes on first run)..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Failed to load model. Please check the error messages above.")
    st.info("""
    **How to fix:**
    1. Go to https://huggingface.co/taibitfd/geocodegpt
    2. Make sure the model is public
    3. Check that all files are uploaded
    4. Refresh this page and try again
    """)
    st.stop()

# Main interface layout
col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area(
        "🌍 Describe what you want to do:",
        height=150,
        placeholder="Example: Create a true color composite of Sentinel-2 for California",
        help="Be specific about the region, dataset, and analysis you want"
    )

with col2:
    st.subheader("💡 Quick Examples")
    examples = [
        "True color composite of Sentinel-2",
        "NDVI for Brazil using Landsat 8",
        "Export Landsat 8 to Drive",
        "NDWI water body detection",
        "Random Forest classification",
        "Time series vegetation health"
    ]
    
    for example in examples:
        full_prompt = f"Create a {example}"
        if st.button(f"📌 {example}", key=example, use_container_width=True):
            prompt = full_prompt
            st.rerun()

# Action buttons
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description of what you want to do.")
    else:
        with st.spinner("🤖 Generating code..."):
            response = generate_code(tokenizer, model, prompt, max_tokens, temperature)
            
            if response.startswith("❌"):
                st.error(response)
            else:
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

# ============================================================
# FOOTER
# ============================================================

st.divider()

# Show generation tips
with st.expander("💡 Tips for better results"):
    st.markdown("""
    - **Be specific** about the region (e.g., "California", "Amazon Rainforest")
    - **Specify the dataset** you want to use (Sentinel-2, Landsat 8, MODIS, etc.)
    - **Describe the analysis** clearly (e.g., "calculate NDVI", "create a time series")
    - **Mention output format** if needed (e.g., "export to Drive", "create a chart")
    - **Include time range** for time series analysis
    """)

# Credits
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")
st.caption(f"⚡ Model loaded from Hugging Face: `{HF_MODEL_NAME}`")

# ============================================================
# CACHE CLEANUP OPTION (Hidden, for debugging)
# ============================================================

if st.sidebar.checkbox("🔄 Clear cache and reload", help="Clear the model cache and reload"):
    st.cache_resource.clear()
    st.success("Cache cleared! Refreshing...")
    time.sleep(1)
    st.rerun()

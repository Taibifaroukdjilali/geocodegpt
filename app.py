# ============================================================
# app.py - GeoCode-GPT from Hugging Face (Cached)
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_NAME = "taibitfd/geocodegpt"

# ============================================================
# MODEL LOADER WITH CACHING
# ============================================================

@st.cache_resource
def load_model():
    """Load model from Hugging Face with caching"""
    try:
        st.info("📥 Loading GeoCode-GPT from Hugging Face...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        st.success("✅ Model loaded successfully from Hugging Face!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("""
        **Troubleshooting:**
        1. Check if the model is public: https://huggingface.co/taibitfd/geocodegpt
        2. Make sure all files are uploaded
        3. Try refreshing the page
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
    
    # Model info
    st.subheader("📊 Model Info")
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("📁 Hosted on Hugging Face Hub")
    st.markdown(f"[🔗 View Model](https://huggingface.co/{HF_MODEL_NAME})")
    
    st.divider()
    st.caption("🔧 Built with ❤️ using Transformers & Streamlit")

# Load model
with st.spinner("🔄 Loading model from Hugging Face (this may take 2-3 minutes on first run)..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
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
        "Create a time series chart of vegetation health"
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
                response = generate_code(tokenizer, model, prompt, max_tokens, temperature)
                
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
                
            except Exception as e:
                st.error(f"❌ Error generating code: {e}")
                st.info("Try adjusting the temperature or max tokens settings.")

# Footer
st.divider()
st.caption("📌 Tip: Generated code is for Google Earth Engine JavaScript API")

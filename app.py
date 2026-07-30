# ============================================================
# app.py - FIXED GENERATION
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import gc

# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_NAME = "taibitfd/geocodegpt"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model():
    """Load model from Hugging Face"""
    try:
        st.info("📥 Loading model from Hugging Face...")
        
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_NAME,
            torch_dtype=torch.float32,  # Use CPU mode
            low_cpu_mem_usage=True
        )
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ============================================================
# GENERATION FUNCTION - SIMPLIFIED AND RELIABLE
# ============================================================

def generate_code(tokenizer, model, prompt, max_tokens, temperature):
    """Generate code with proper error handling"""
    try:
        # Simple system prompt
        system = "You are a Google Earth Engine expert. Generate only JavaScript code. No markdown or explanations."
        full_prompt = f"{system}\n\nUser request: {prompt}\n\nCODE:"
        
        # Tokenize with shorter length
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            max_length=1024,  # Reduced from 2048
            truncation=True
        )
        
        # Move to CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Generate with shorter output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),  # Limit to 512 tokens
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        elif "```javascript" in response:
            response = response.split("```javascript")[-1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        
        return response
        
    except Exception as e:
        return f"Error: {str(e)}"

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

# Load model
tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("❌ Failed to load model.")
    st.stop()

# Main interface
prompt = st.text_area(
    "🌍 Describe what you want to do:",
    height=120,
    placeholder="Example: Create a true color composite of Sentinel-2 for California"
)

# Example buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🛰️ NDVI", use_container_width=True):
        prompt = "Calculate NDVI using Sentinel-2 for California"
        st.rerun()
with col2:
    if st.button("🌿 True Color", use_container_width=True):
        prompt = "Create a true color composite of Sentinel-2"
        st.rerun()
with col3:
    if st.button("💧 Water Detection", use_container_width=True):
        prompt = "Create NDWI water body detection using Sentinel-2"
        st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("🤖 Generating code..."):
            # Show progress with a placeholder
            progress_placeholder = st.empty()
            progress_placeholder.info("⏳ Processing your request... This may take 10-20 seconds.")
            
            try:
                # Generate code
                start_time = time.time()
                response = generate_code(tokenizer, model, prompt, max_tokens, temperature)
                elapsed_time = time.time() - start_time
                
                progress_placeholder.empty()
                
                if response.startswith("Error"):
                    st.error(response)
                else:
                    st.subheader(f"💻 Generated Code (took {elapsed_time:.1f}s)")
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
                progress_placeholder.empty()
                st.error(f"❌ Error generating code: {str(e)}")
                st.info("""
                **Troubleshooting:**
                1. The model might be processing - wait 30 seconds
                2. Try simplifying your request
                3. Reduce the max tokens
                4. Refresh the page and try again
                """)
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Footer
st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")

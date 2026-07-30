# ============================================================
# app.py - GeoCode-GPT with Hugging Face Inference API
# ============================================================

import streamlit as st
import requests
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================

# Your Hugging Face model and token
HF_MODEL_NAME = "taibitfd/geocodegpt"
HF_TOKEN = "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"

# API URL for your model
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def query_huggingface_api(prompt, max_tokens=512, temperature=0.7):
    """Send request to Hugging Face Inference API"""
    
    # Prepare the prompt
    system = """You are a Google Earth Engine expert. Generate only JavaScript code. 
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
    full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
    
    # API payload
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.95,
            "return_full_text": False
        }
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # Send request to Hugging Face
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                # Clean up the response
                if "CODE:" in generated_text:
                    generated_text = generated_text.split("CODE:")[-1].strip()
                elif "```javascript" in generated_text:
                    generated_text = generated_text.split("```javascript")[-1].split("```")[0].strip()
                elif "```" in generated_text:
                    generated_text = generated_text.split("```")[1].strip()
                return generated_text
            else:
                return "Error: Unexpected response format"
        elif response.status_code == 503:
            return "⏳ Model is loading (cold start). Please wait 30 seconds and try again."
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. The model might be busy. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
# UI
# ============================================================

st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("Generate Google Earth Engine JavaScript code using AI via Hugging Face API")

# Sidebar
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
    st.subheader("📊 Model Info")
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("⚡ Using Hugging Face Inference API")
    st.caption("💡 No local GPU required!")
    st.markdown(f"[🔗 View on HF](https://huggingface.co/{HF_MODEL_NAME})")
    
    st.divider()
    st.caption("🔧 Built with ❤️ using Streamlit")

# Main interface
prompt = st.text_area(
    "🌍 Describe what you want to do:",
    height=120,
    placeholder="Example: Calculate NDVI using Sentinel-2 for California"
)

# Quick example buttons
st.subheader("💡 Quick Examples")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🛰️ NDVI", use_container_width=True):
        prompt = "Calculate NDVI using Sentinel-2 for California"
        st.rerun()

with col2:
    if st.button("🌿 True Color", use_container_width=True):
        prompt = "Create a true color composite of Sentinel-2 for California"
        st.rerun()

with col3:
    if st.button("💧 Water Detection", use_container_width=True):
        prompt = "Create NDWI water body detection using Sentinel-2"
        st.rerun()

with col4:
    if st.button("📤 Export", use_container_width=True):
        prompt = "Export an image to Google Drive"
        st.rerun()

# More examples in expander
with st.expander("📋 More Examples"):
    example_prompts = [
        "Generate a land cover classification using Random Forest",
        "Create a time series chart of vegetation health",
        "Detect deforestation using Sentinel-2 time series",
        "Calculate EVI using MODIS data",
        "Create a median composite of Landsat 8 for a region",
        "Calculate NDWI for a lake using Sentinel-2",
        "Add a band to an image collection using a custom function"
    ]
    
    for ex in example_prompts:
        if st.button(ex, use_container_width=True):
            prompt = ex
            st.rerun()

# Generate button
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description of what you want to do.")
    else:
        # Create a progress container
        progress_container = st.empty()
        progress_container.info("⏳ Sending request to Hugging Face API...")
        
        with st.spinner("🤖 Generating code..."):
            start_time = time.time()
            response = query_huggingface_api(prompt, max_tokens, temperature)
            elapsed = time.time() - start_time
            
            progress_container.empty()
            
            # Check for errors
            if response.startswith("Error") or response.startswith("API Error"):
                st.error(response)
                
                if "503" in response or "cold start" in response:
                    st.info("""
                    **💡 The model is waking up (cold start).** 
                    This happens on first use or after inactivity.
                    - Wait 30 seconds and click "Generate" again
                    - The second attempt will be much faster
                    """)
                elif "timeout" in response.lower():
                    st.info("""
                    **💡 Request timed out.**
                    - Try reducing max tokens
                    - Try a simpler prompt
                    - Wait a moment and try again
                    """)
            else:
                # Success!
                st.success(f"✅ Code generated in {elapsed:.1f} seconds!")
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
                    if st.button("📋 Copy to Clipboard", use_container_width=True):
                        st.write("Select all and copy!")

# Footer
st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")
st.caption("⚡ Powered by Hugging Face Inference API (free tier)")

# Debug info (only visible in development)
with st.expander("🔍 Debug Info"):
    st.caption(f"Model: {HF_MODEL_NAME}")
    st.caption(f"API URL: {API_URL}")
    st.caption(f"Token: {'✅ Set' if HF_TOKEN else '❌ Missing'}")

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

HF_MODEL_NAME = "taibitfd/geocodegpt"
HF_TOKEN = "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"

# API URL
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# INFERENCE FUNCTION WITH RETRY LOGIC
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
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Send request to Hugging Face
            response = requests.post(
                API_URL, 
                json=payload, 
                headers=headers, 
                timeout=90
            )
            
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
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait 5 seconds before retry
                    continue
                return "⏳ Model is loading (cold start). Please wait 30 seconds and try again."
                
            else:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"API Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return "⏰ Request timed out. Please try again."
            
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return "🔌 Connection error. Please check your internet."
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error: {str(e)}"
    
    return "Error: All retries failed"

# ============================================================
# UI
# ============================================================

st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("Generate Google Earth Engine JavaScript code using AI")

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
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("⚡ Using Hugging Face Inference API")
    st.caption("💡 No local GPU required!")
    
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
    if st.button("💧 Water", use_container_width=True):
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
        "Add a band to an image collection using a custom function"
    ]
    
    for ex in example_prompts:
        if st.button(ex, use_container_width=True):
            prompt = ex
            st.rerun()

# Generate button
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        # Create a progress container
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Sending request to Hugging Face API... This may take 10-30 seconds.")
        
        with st.spinner("🤖 Generating code..."):
            start_time = time.time()
            response = query_huggingface_api(prompt, max_tokens, temperature)
            elapsed = time.time() - start_time
            
            status_placeholder.empty()
            
            # Check for errors
            if response.startswith("Error") or response.startswith("API Error"):
                st.error(response)
                
                if "503" in response or "cold start" in response:
                    st.info("""
                    **💡 The model is waking up (cold start).** 
                    - Wait 30 seconds and try again
                    - The second attempt will be faster
                    """)
                elif "timeout" in response.lower():
                    st.info("""
                    **💡 Request timed out.**
                    - Try reducing max tokens
                    - Try a simpler prompt
                    - Wait and try again
                    """)
                elif "connection" in response.lower():
                    st.info("""
                    **💡 Connection issue.**
                    - This is usually temporary
                    - Wait a moment and try again
                    - Refresh the page
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
                        st.write("Select the code and press Ctrl+C to copy!")

# Footer
st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")
st.caption("⚡ Powered by Hugging Face Inference API")

# Debug info
with st.expander("🔍 Status"):
    st.caption(f"Model: {HF_MODEL_NAME}")
    st.caption(f"API URL: {API_URL}")
    st.caption(f"Token: {'✅ Set' if HF_TOKEN else '❌ Missing'}")
    st.caption(f"Streamlit Version: {st.__version__}")

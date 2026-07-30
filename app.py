# ============================================================
# app.py - Using Alternative Hugging Face Endpoint
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

# Try different endpoints
ENDPOINTS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_NAME}",
    f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
]

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# INFERENCE FUNCTION WITH MULTIPLE ENDPOINTS
# ============================================================

def query_huggingface_api(prompt, max_tokens=512, temperature=0.7):
    """Send request to Hugging Face Inference API with multiple endpoints"""
    
    # Prepare the prompt
    system = """You are a Google Earth Engine expert. Generate only JavaScript code. 
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
    full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
    
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
    
    # Try each endpoint
    for endpoint in ENDPOINTS:
        try:
            response = requests.post(
                endpoint, 
                json=payload, 
                headers=headers, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    if generated_text:
                        # Clean up
                        if "CODE:" in generated_text:
                            generated_text = generated_text.split("CODE:")[-1].strip()
                        elif "```javascript" in generated_text:
                            generated_text = generated_text.split("```javascript")[-1].split("```")[0].strip()
                        elif "```" in generated_text:
                            generated_text = generated_text.split("```")[1].strip()
                        return generated_text
                    else:
                        continue
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', '')
                    if generated_text:
                        return generated_text
                    else:
                        continue
                else:
                    continue
                    
            elif response.status_code == 503:
                return "⏳ Model is loading. Please wait 30 seconds and try again."
                
            elif response.status_code == 429:
                return "⏳ Rate limit exceeded. Please wait a moment and try again."
                
            else:
                continue  # Try next endpoint
                
        except requests.exceptions.ConnectionError:
            continue  # Try next endpoint
        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue
    
    return "⚠️ All endpoints failed. Please try again later."

# ============================================================
# UI
# ============================================================

st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("Generate Google Earth Engine JavaScript code using AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 128, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("💡 Using fallback endpoints")

# Main interface
prompt = st.text_area(
    "🌍 Describe what you want to do:",
    height=120,
    placeholder="Example: Calculate NDVI using Sentinel-2 for California"
)

# Quick examples
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
    if st.button("💧 Water", use_container_width=True):
        prompt = "Create NDWI water body detection using Sentinel-2"
        st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("🤖 Generating code..."):
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Sending request to Hugging Face API...")
            
            start_time = time.time()
            response = query_huggingface_api(prompt, max_tokens, temperature)
            elapsed = time.time() - start_time
            
            status_placeholder.empty()
            
            if response.startswith("⏳") or response.startswith("⚠️"):
                st.warning(response)
                if "cold start" in response:
                    st.info("💡 Wait 30 seconds and try again")
            else:
                st.success(f"✅ Code generated in {elapsed:.1f} seconds!")
                st.subheader("💻 Generated Code")
                st.code(response, language="javascript")
                
                st.download_button(
                    label="📥 Download Code",
                    data=response,
                    file_name="earth_engine_code.js",
                    mime="text/javascript",
                    use_container_width=True
                )

# Footer
st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")

# ============================================================
# app.py - GeoCode-GPT with Robust API Handling
# ============================================================

import streamlit as st
import requests
import json
import time
import socket
import dns.resolver

# ============================================================
# CONFIGURATION
# ============================================================

HF_MODEL_NAME = "taibitfd/geocodegpt"
HF_TOKEN = "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"

# Try different API endpoints
API_ENDPOINTS = [
    f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
    f"https://api-inference.huggingface.co/v1/models/{HF_MODEL_NAME}",
    f"https://huggingface.co/api/models/{HF_MODEL_NAME}/inference",
]

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_network():
    """Check if we can reach Hugging Face"""
    try:
        socket.gethostbyname('api-inference.huggingface.co')
        return True
    except:
        return False

def query_with_retry(prompt, max_tokens=512, temperature=0.7, max_retries=3):
    """Send request with retry logic"""
    
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
    
    # Try multiple endpoints
    for endpoint in API_ENDPOINTS:
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    endpoint, 
                    json=payload, 
                    headers=headers, 
                    timeout=60,
                    verify=True
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        # Clean up
                        if "CODE:" in generated_text:
                            generated_text = generated_text.split("CODE:")[-1].strip()
                        elif "```javascript" in generated_text:
                            generated_text = generated_text.split("```javascript")[-1].split("```")[0].strip()
                        elif "```" in generated_text:
                            generated_text = generated_text.split("```")[1].strip()
                        return generated_text
                    return "Error: Unexpected response format"
                    
                elif response.status_code == 503:
                    return "⏳ Model is loading (cold start). Please wait 30 seconds and try again."
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return f"API Error: {response.status_code} - {response.text}"
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "⏰ Request timed out. Please try again."
                
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "🔌 Connection error. Please check your internet."
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return f"Error: {str(e)}"
    
    return "Error: All endpoints failed"

# ============================================================
# UI
# ============================================================

st.title("🌍 GeoCode-GPT - Earth Engine Code Generator")
st.markdown("Generate Google Earth Engine JavaScript code using AI")

# Check network status
if not check_network():
    st.warning("⚠️ Network issue detected. Using alternative method...")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 128, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("⚡ Using Hugging Face Inference API")
    
    # Network status
    network_ok = check_network()
    if network_ok:
        st.success("✅ Network OK")
    else:
        st.error("❌ Network Issue")

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
    if st.button("💧 Water Detection", use_container_width=True):
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
            response = query_with_retry(prompt, max_tokens, temperature)
            elapsed = time.time() - start_time
            
            status_placeholder.empty()
            
            if response.startswith("Error") or response.startswith("API Error"):
                st.error(response)
                st.info("""
                **💡 Troubleshooting:**
                1. Wait a moment and try again
                2. The model might be waking up (cold start)
                3. Try a simpler prompt
                4. Reduce max tokens
                """)
                
                # Add a manual retry button
                if st.button("🔄 Retry Now"):
                    st.rerun()
                    
            elif "cold start" in response.lower():
                st.warning(response)
                st.info("💡 Wait 30 seconds and click 'Generate' again")
                
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

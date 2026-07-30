# ============================================================
# app.py - GeoCode-GPT with Proper Response Handling
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
# INFERENCE FUNCTION WITH PROPER RESPONSE HANDLING
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
        response = requests.post(
            API_URL, 
            json=payload, 
            headers=headers, 
            timeout=120
        )
        
        # Check response status
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Standard response format
                generated_text = result[0].get('generated_text', '')
                if generated_text:
                    # Clean up the response
                    if "CODE:" in generated_text:
                        generated_text = generated_text.split("CODE:")[-1].strip()
                    elif "```javascript" in generated_text:
                        generated_text = generated_text.split("```javascript")[-1].split("```")[0].strip()
                    elif "```" in generated_text:
                        generated_text = generated_text.split("```")[1].strip()
                    
                    # Check if we got actual code
                    if generated_text and len(generated_text) > 10:
                        return generated_text
                    else:
                        return "⚠️ Generated response was empty. Please try again with a different prompt."
                else:
                    return "⚠️ No text generated. Please try again."
            
            elif isinstance(result, dict):
                # Some models return a dict
                generated_text = result.get('generated_text', '')
                if generated_text:
                    if "CODE:" in generated_text:
                        generated_text = generated_text.split("CODE:")[-1].strip()
                    return generated_text
                else:
                    return "⚠️ No text generated. Please try again."
            
            else:
                return f"⚠️ Unexpected response format: {str(result)[:200]}"
                
        elif response.status_code == 503:
            return "⏳ Model is loading (cold start). Please wait 30 seconds and try again."
            
        elif response.status_code == 429:
            return "⏳ Rate limit exceeded. Please wait a moment and try again."
            
        else:
            return f"⚠️ API Error: {response.status_code} - {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please try again."
        
    except requests.exceptions.ConnectionError:
        return "🔌 Connection error. Please check your internet."
        
    except json.JSONDecodeError:
        return f"⚠️ Invalid JSON response: {response.text[:200]}"
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

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
        step=64
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    
    st.divider()
    st.caption(f"🤗 Model: `{HF_MODEL_NAME}`")
    st.caption("⚡ Using Hugging Face Inference API")

# Main interface
prompt = st.text_area(
    "🌍 Describe what you want to do:",
    height=120,
    placeholder="Example: Calculate NDVI using Sentinel-2 for California"
)

# Quick example buttons
st.subheader("💡 Quick Examples")
col1, col2, col3 = st.columns(3)

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

# Generate button
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Sending request to Hugging Face API...")
        
        with st.spinner("🤖 Generating code..."):
            start_time = time.time()
            response = query_huggingface_api(prompt, max_tokens, temperature)
            elapsed = time.time() - start_time
            
            status_placeholder.empty()
            
            # Display response
            if response.startswith("⏳") or response.startswith("⏰") or response.startswith("🔌"):
                st.warning(response)
                if "cold start" in response:
                    st.info("💡 Wait 30 seconds and click 'Generate' again")
                elif "rate limit" in response:
                    st.info("💡 Wait a moment and try again")
                    
            elif response.startswith("⚠️"):
                st.error(response)
                st.info("💡 Try a simpler prompt or reduce max tokens")
                
            else:
                # Success - show the code
                st.success(f"✅ Code generated in {elapsed:.1f} seconds!")
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

# Footer
st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")

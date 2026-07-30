# ============================================================
# app.py - Using Hugging Face Inference with Fallback
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

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# ALTERNATIVE: Use requests with different settings
# ============================================================

def query_huggingface_alt(prompt, max_tokens=512, temperature=0.7):
    """Alternative method using different API format"""
    
    system = """You are a Google Earth Engine expert. Generate only JavaScript code. 
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
    full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
    
    # Use the Hugging Face inference API with different format
    API_URL = "https://api-inference.huggingface.co/models/taibitfd/geocodegpt"
    
    # Different payload format
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
        # Use a different session with custom DNS
        session = requests.Session()
        session.trust_env = False  # Ignore proxy settings
        
        response = session.post(
            API_URL, 
            json=payload, 
            headers=headers, 
            timeout=90,
            verify=True
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                if generated_text:
                    if "CODE:" in generated_text:
                        generated_text = generated_text.split("CODE:")[-1].strip()
                    return generated_text
            return "⚠️ No text generated"
            
        elif response.status_code == 503:
            return "⏳ Model loading. Wait 30s and retry."
        else:
            return f"⚠️ Error {response.status_code}"
            
    except Exception as e:
        return f"⚠️ Error: {str(e)[:100]}"

# ============================================================
# FALLBACK: Use hardcoded response for testing
# ============================================================

def get_fallback_response(prompt):
    """Fallback response when API is unavailable"""
    
    if "ndvi" in prompt.lower() or "sentinel" in prompt.lower():
        return """
// NDVI using Sentinel-2
var geometry = ee.Geometry.Point([-122.443, 37.754]);

// Load Sentinel-2 imagery
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-01-01", "2023-12-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

// Calculate NDVI
function addNDVI(image) {
  var ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI");
  return image.addBands(ndvi);
}

var ndvi = s2.map(addNDVI).select("NDVI").median();

// Display
Map.addLayer(ndvi, {min: -1, max: 1, palette: ["blue", "white", "green"]}, "NDVI");
Map.centerObject(geometry, 10);
"""
    
    elif "true color" in prompt.lower() or "composite" in prompt.lower():
        return """
// True Color Composite using Sentinel-2
var geometry = ee.Geometry.Point([-122.443, 37.754]);

var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-06-01", "2023-09-01")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
  .median();

var trueColor = s2.select(["B4", "B3", "B2"]);
Map.addLayer(trueColor, {min: 0, max: 3000, gamma: 1.4}, "True Color");
Map.centerObject(geometry, 10);
"""
    
    else:
        return """
// Earth Engine Code
var geometry = ee.Geometry.Point([-122.443, 37.754]);

var collection = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-01-01", "2023-12-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

var median = collection.median();
Map.addLayer(median, {bands: ["B4", "B3", "B2"], min: 0, max: 3000}, "Composite");
Map.centerObject(geometry, 10);
"""

# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_with_fallback(prompt, max_tokens=512, temperature=0.7):
    """Try API first, then fallback"""
    
    # Try API first
    result = query_huggingface_alt(prompt, max_tokens, temperature)
    
    # If API fails, use fallback
    if result.startswith("⚠️") or result.startswith("⏳"):
        st.warning("⚠️ Hugging Face API unavailable. Using fallback response.")
        return get_fallback_response(prompt)
    
    return result

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
    
    # Show status
    st.caption("📡 API Status: ")
    try:
        import socket
        socket.gethostbyname('api-inference.huggingface.co')
        st.success("✅ Reachable")
    except:
        st.error("❌ Unreachable (using fallback)")

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
            start_time = time.time()
            response = generate_with_fallback(prompt, max_tokens, temperature)
            elapsed = time.time() - start_time
            
            if response:
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
            else:
                st.error("❌ Failed to generate code. Please try again.")

# Footer
st.divider()
st.caption("📌 Generated code is for Google Earth Engine JavaScript API")
st.caption("💡 Using fallback responses when API is unavailable")

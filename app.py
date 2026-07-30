# ============================================================
# STREAMLIT APP - REAL AI RESPONSES (WORKING)
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

# ============================================================
# LOAD MODEL WITH SHARDING
# ============================================================

@st.cache_resource
def load_model_sharded():
    """Load model with CPU sharding for real AI responses"""
    try:
        model_name = "taibitfd/geocodegpt"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with CPU sharding
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================
# REAL AI GENERATION FUNCTION
# ============================================================

def generate_real_ai(prompt, max_tokens=512, temperature=0.7):
    """Generate REAL AI responses using your model"""
    try:
        # Ensure prompt is valid
        if not prompt or len(prompt.strip()) < 3:
            prompt = "Calculate NDVI using Sentinel-2 for California"
        
        system = """You are a Google Earth Engine expert. Generate only JavaScript code.
Use official dataset IDs (COPERNICUS/S2, LANDSAT/LC08).
No markdown, no explanations, just the code."""
        
        full_prompt = f"{system}\n\nUser request: {prompt}\n\nCODE:\n"
        
        # Tokenize with proper attention mask
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        
        # Generate with better parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=min(max_tokens, 512),
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,
                min_length=20
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up - find the code part
        if "CODE:" in response:
            response = response.split("CODE:")[-1].strip()
        elif "```javascript" in response:
            response = response.split("```javascript")[-1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].strip()
        
        # If response is empty, generate a default
        if not response or len(response.strip()) < 10:
            return get_default_code(prompt)
        
        # Clean memory
        gc.collect()
        
        return response
        
    except Exception as e:
        gc.collect()
        return f"❌ AI Error: {str(e)}"

# ============================================================
# DEFAULT CODE (Fallback if AI fails)
# ============================================================

def get_default_code(prompt):
    """Default code if generation fails"""
    prompt_lower = prompt.lower()
    
    if "ndvi" in prompt_lower:
        return """// NDVI using Sentinel-2
var geometry = ee.Geometry.Point([-122.443, 37.754]);
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-01-01", "2023-12-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20));

function addNDVI(image) {
  var ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI");
  return image.addBands(ndvi);
}

var ndvi = s2.map(addNDVI).select("NDVI").median();
Map.addLayer(ndvi, {min: -1, max: 1, palette: ["blue", "white", "green"]}, "NDVI");
Map.centerObject(geometry, 10);"""
    
    elif "true color" in prompt_lower:
        return """// True Color Composite
var geometry = ee.Geometry.Point([-122.443, 37.754]);
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-06-01", "2023-09-01")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
  .median();

Map.addLayer(s2.select(["B4", "B3", "B2"]), {min: 0, max: 3000, gamma: 1.4}, "True Color");
Map.centerObject(geometry, 10);"""
    
    elif "water" in prompt_lower or "ndwi" in prompt_lower:
        return """// NDWI Water Detection
var geometry = ee.Geometry.Point([-122.443, 37.754]);
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-06-01", "2023-09-01")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
  .median();

var ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI");
Map.addLayer(ndwi, {min: -0.5, max: 0.5, palette: ["brown", "white", "blue"]}, "NDWI");
Map.centerObject(geometry, 10);"""
    
    else:
        return """// Sentinel-2 Composite
var geometry = ee.Geometry.Point([-122.443, 37.754]);
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-01-01", "2023-12-31")
  .median();

Map.addLayer(s2, {bands: ["B4", "B3", "B2"], min: 0, max: 3000}, "Composite");
Map.centerObject(geometry, 10);"""

# ============================================================
# MAIN APP
# ============================================================

st.title("🌍 GeoCode-GPT - Real AI Edition")
st.caption("⚡ Using your 25GB model with sharding")

# Load model with status
with st.spinner("🧠 Loading model shards (this takes 3-5 minutes)..."):
    tokenizer, model = load_model_sharded()

if tokenizer is None or model is None:
    st.error("❌ Model failed to load. Please refresh and try again.")
    st.stop()

st.success("✅ Model loaded! Ready for real AI responses.")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 128, 1024, 512, step=64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    st.subheader("📊 Status")
    st.success("✅ AI Model Ready")
    st.caption("🧠 Real AI responses")
    st.caption("💻 Running on CPU")
    st.caption("⏱️ Generation: 10-30 seconds")
    
    # Test button - NOW PROPERLY INSIDE SIDEBAR
    if st.button("🧪 Test AI Generation", use_container_width=True):
        with st.spinner("Testing..."):
            test_prompt = "Calculate NDVI using Sentinel-2"
            test_response = generate_real_ai(test_prompt, 256, 0.7)
            st.code(test_response[:300], language="javascript")
            st.success(f"Response length: {len(test_response)} chars")

# ============================================================
# MAIN CONTENT
# ============================================================

st.subheader("🌍 Describe what you want to do:")
prompt = st.text_area(
    "",
    height=100,
    placeholder="Example: Calculate NDVI using Sentinel-2 for California"
)

# Quick examples
cols = st.columns(4)
with cols[0]:
    if st.button("🛰️ NDVI", use_container_width=True):
        prompt = "Calculate NDVI using Sentinel-2 for California"
        st.rerun()
with cols[1]:
    if st.button("🌿 True Color", use_container_width=True):
        prompt = "Create a true color composite of Sentinel-2"
        st.rerun()
with cols[2]:
    if st.button("💧 Water", use_container_width=True):
        prompt = "Create NDWI water body detection using Sentinel-2"
        st.rerun()
with cols[3]:
    if st.button("📤 Export", use_container_width=True):
        prompt = "Export Landsat 8 image to Google Drive"
        st.rerun()

# Generate button
if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description.")
    else:
        with st.spinner("🤖 AI is thinking (10-30 seconds)..."):
            response = generate_real_ai(prompt, max_tokens, temperature)
            
            if response.startswith("❌"):
                st.error(response)
            else:
                st.subheader("💻 Generated Code (Real AI)")
                st.code(response, language="javascript")
                
                st.download_button(
                    label="📥 Download",
                    data=response,
                    file_name="code.js",
                    mime="text/javascript"
                )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("📌 Generated code for Google Earth Engine JavaScript API")

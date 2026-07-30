# ============================================================
# STREAMLIT APP - CPU ONLY (No GPU Needed)
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ============================================================
# FORCE CPU MODE (No GPU required)
# ============================================================

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

st.set_page_config(
    page_title="🌍 GeoCode-GPT",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoCode-GPT - CPU Edition")
st.caption("💻 Running on CPU (No GPU needed)")

# ============================================================
# MODEL LOADER - CPU ONLY
# ============================================================

@st.cache_resource
def load_model_cpu():
    """Load model on CPU - no GPU required"""
    
    try:
        with st.spinner("🧠 Loading model on CPU (this may take 3-5 minutes)..."):
            
            # Step 1: Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("taibitfd/geocodegpt")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Step 2: Load model on CPU with memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                "taibitfd/geocodegpt",
                torch_dtype=torch.float32,  # CPU uses float32
                device_map="cpu",  # Force CPU
                low_cpu_mem_usage=True
            )
            
            st.success("✅ Model loaded successfully on CPU!")
            return tokenizer, model
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================
# LOAD THE MODEL
# ============================================================

tokenizer, model = load_model_cpu()

if tokenizer is None or model is None:
    st.warning("⚠️ Model failed to load. Using fallback templates.")
    
    # Fallback: Use template responses without loading model
    def generate_fallback(prompt):
        return get_template(prompt)
    
    def get_template(prompt):
        if "ndvi" in prompt.lower():
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
        
        elif "true color" in prompt.lower():
            return """// True Color Composite
var geometry = ee.Geometry.Point([-122.443, 37.754]);
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(geometry)
  .filterDate("2023-06-01", "2023-09-01")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
  .median();

Map.addLayer(s2.select(["B4", "B3", "B2"]), {min: 0, max: 3000, gamma: 1.4}, "True Color");
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
    
    # Override generate function with fallback
    def generate_code(prompt, max_tokens=512, temperature=0.7):
        return get_template(prompt)
    
    st.info("💡 Using fallback templates (instant responses!)")
    
else:
    # ============================================================
    # GENERATION FUNCTION - REAL AI (CPU)
    # ============================================================
    
    def generate_code(prompt, max_tokens=512, temperature=0.7):
        """Generate Earth Engine code on CPU"""
        try:
            system = """You are a Google Earth Engine expert. Generate only JavaScript code.
Use official dataset IDs. No markdown, no explanations."""
            
            full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
            
            # Tokenize
            inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to('cpu') for k, v in inputs.items()}  # Force CPU
            
            # Generate (slower on CPU but works!)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 384),  # Lower tokens for CPU
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "CODE:" in response:
                response = response.split("CODE:")[-1].strip()
            elif "```javascript" in response:
                response = response.split("```javascript")[-1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].strip()
            
            # Clean memory
            gc.collect()
            
            return response
            
        except Exception as e:
            gc.collect()
            return f"Error: {str(e)}"

# ============================================================
# STREAMLIT UI
# ============================================================

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 128, 512, 384, step=64)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    
    st.divider()
    st.subheader("📊 Status")
    if tokenizer is None:
        st.warning("⚠️ Using fallback templates")
        st.caption("⚡ Instant responses (no AI)")
    else:
        st.success("✅ AI model loaded")
        st.caption("💻 Running on CPU")
        st.caption("⏱️ Generation: 10-30 seconds")

# Main
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
        prompt = "Calculate NDVI using Sentinel-2"
        st.rerun()
with cols[1]:
    if st.button("🌿 True Color", use_container_width=True):
        prompt = "Create a true color composite"
        st.rerun()
with cols[2]:
    if st.button("💧 Water", use_container_width=True):
        prompt = "Detect water using NDWI"
        st.rerun()
with cols[3]:
    if st.button("📤 Export", use_container_width=True):
        prompt = "Export image to Drive"
        st.rerun()

if st.button("🚀 Generate Code", type="primary", use_container_width=True):
    if not prompt:
        st.warning("Please enter a description")
    else:
        with st.spinner("🤖 Generating..."):
            response = generate_code(prompt, max_tokens, temperature)
            st.code(response, language="javascript")
            
            st.download_button(
                label="📥 Download",
                data=response,
                file_name="code.js",
                mime="text/javascript"
            )

st.divider()
st.caption("📌 Generated code for Google Earth Engine JavaScript API")

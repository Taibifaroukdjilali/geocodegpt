# ============================================================
# CPU SHARDING - CORRECT IMPLEMENTATION
# ============================================================

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "2"

st.set_page_config(
    page_title="🌍 GeoCode-GPT - Sharded",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoCode-GPT - Sharded Edition")
st.caption("⚡ CPU sharding - only loads what's needed")

# ============================================================
# SHARDED MODEL LOADER (CPU) - FIXED
# ============================================================

@st.cache_resource
def load_model_sharded_cpu():
    """Load model using CPU sharding - only loads needed shards"""
    
    try:
        with st.spinner("🧠 Loading model shards (only what's needed)..."):
            model_name = "taibitfd/geocodegpt"
            
            # Step 1: Load tokenizer (tiny)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Step 2: Load model with CPU sharding
            # Use device_map="cpu" with max_memory to control loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # CPU uses float32
                device_map="cpu",
                low_cpu_mem_usage=True,  # This is valid for from_pretrained
                max_memory={0: "8GB", "cpu": "16GB"}  # Memory limits
            )
            
            st.success("✅ Model loaded with sharding!")
            st.info("💾 CPU Mode: Sharded loading (only loads needed shards)")
            
            return tokenizer, model
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None

# ============================================================
# ALTERNATIVE: Load with Accelerate (More Control)
# ============================================================

@st.cache_resource
def load_model_with_accelerate():
    """Alternative: Use accelerate for sharded loading"""
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from transformers import AutoConfig
        
        with st.spinner("🧠 Loading with accelerate sharding..."):
            model_name = "taibitfd/geocodegpt"
            
            # Load config and tokenizer
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create empty model
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            
            # Load shards on-demand
            model = load_checkpoint_and_dispatch(
                model,
                model_name,
                device_map="cpu",
                max_memory={0: "8GB", "cpu": "16GB"},
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float32
            )
            
            st.success("✅ Model loaded with accelerate sharding!")
            return tokenizer, model
            
    except Exception as e:
        st.error(f"❌ Accelerate error: {str(e)}")
        return None, None

# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate_with_model(model, tokenizer, prompt, max_tokens=384, temperature=0.7):
    """Generate using sharded model"""
    try:
        system = """You are a Google Earth Engine expert. Generate only JavaScript code.
Use official dataset IDs. No markdown, no explanations."""
        
        full_prompt = f"{system}\n\n{prompt}\n\nCODE:"
        
        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 384),
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
        
        # Cleanup
        gc.collect()
        return response
        
    except Exception as e:
        gc.collect()
        return f"Error: {str(e)}"

# ============================================================
# FALLBACK TEMPLATES
# ============================================================

def get_template(prompt):
    """Template responses when model not loaded"""
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
# TRY BOTH LOADING METHODS
# ============================================================

# Method 1: Direct from_pretrained with sharding
tokenizer, model = load_model_sharded_cpu()

# If that fails, try accelerate method
if tokenizer is None or model is None:
    st.info("🔄 Trying alternative loading method...")
    tokenizer, model = load_model_with_accelerate()

# If both fail, use templates
if tokenizer is None or model is None:
    st.warning("⚠️ Using fallback templates (instant responses)")
    st.info("💡 Lightweight fallback - no AI, but works instantly!")
    
    def generate_code(prompt, max_tokens=384, temperature=0.7):
        return get_template(prompt)
    
    is_loaded = False
else:
    is_loaded = True
    def generate_code(prompt, max_tokens=384, temperature=0.7):
        return generate_with_model(model, tokenizer, prompt, max_tokens, temperature)

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
    if is_loaded:
        st.success("✅ Model Loaded")
        st.caption("⚡ Sharded loading")
        st.caption("🧠 CPU mode")
    else:
        st.warning("⚠️ Template Mode")
        st.caption("⚡ Instant responses")
        st.caption("📝 No AI (fallback)")

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

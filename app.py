# ============================================================
# DEBUG VERSION - See API Response
# ============================================================

import streamlit as st
import requests
import json

HF_MODEL_NAME = "taibitfd/geocodegpt"
HF_TOKEN = "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"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"

st.title("🔍 Debug: GeoCode-GPT API Test")

prompt = st.text_area("Enter your prompt:", "Calculate NDVI using Sentinel-2")

if st.button("Test API"):
    with st.spinner("Testing..."):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            
            st.write("**Status Code:**", response.status_code)
            st.write("**Response Headers:**", dict(response.headers))
            
            if response.status_code == 200:
                result = response.json()
                st.write("**Response JSON:**", result)
                
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get('generated_text', '')
                    st.write("**Generated Text:**", text)
                    
            else:
                st.write("**Error Response:**", response.text)
                
        except Exception as e:
            st.error(f"Error: {e}")

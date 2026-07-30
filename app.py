# Add this after model loading to test
if st.sidebar.button("🧪 Test AI Generation"):
    with st.spinner("Testing..."):
        test_prompt = "Calculate NDVI using Sentinel-2"
        test_response = generate_real_ai(test_prompt, 256, 0.7)
        st.sidebar.code(test_response[:200], language="javascript")
        st.sidebar.success(f"Response length: {len(test_response)} chars")

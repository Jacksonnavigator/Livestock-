import streamlit as st
from PIL import Image
import requests

# Title and description
st.title("Livestock Health Analyzer: AI-Powered Disease Diagnosis")
st.write("""
    Upload an image of your livestock, and the system will diagnose potential diseases and suggest management options.
""")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Call the backend API (assuming it's already set up)
    st.write("Analyzing the image...")
    url = "http://your-backend-api-url/predict"  # Replace with your backend API URL
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        result = response.json()
        st.write("Diagnosis:", result["diagnosis"])
        st.write("Recommended Action:", result["recommendation"])
    else:
        st.write("Error:", response.text)

# For local testing, you can run `app.py` with `streamlit run app.py`

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from model import load_model, predict

# Load the model
model = load_model()

st.title("Livestock Health Analyzer")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = predict(model, img_array)
    
    st.write(f"Prediction: {prediction}")

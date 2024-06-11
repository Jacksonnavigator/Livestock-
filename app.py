import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pre-trained model (replace 'model.h5' with the path to your model)
# model = load_model('path_to_your_model.h5')

def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(image):
    # Resize image to the input size required by the model
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize to [0, 1] range
    img = np.expand_dims(img, axis=0)
    return img

def predict_disease(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Predict using the model (placeholder code)
    # predictions = model.predict(processed_image)
    
    # Dummy prediction (replace with actual prediction code)
    predictions = [0.6, 0.2, 0.1, 0.1]  # Example probabilities for four classes
    classes = ['Disease A', 'Disease B', 'Disease C', 'Healthy']
    predicted_class = classes[np.argmax(predictions)]
    
    return predicted_class, predictions

def main():
    st.title("Livestock Health Analyzer: AI-Powered Disease Diagnosis")
    
    st.subheader("Upload an image of your livestock to get a health diagnosis.")
    
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if image_file is not None:
        # Load and display the image
        image = load_image(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Perform prediction
        if st.button("Analyze"):
            predicted_class, predictions = predict_disease(image)
            
            st.success(f"Prediction: {predicted_class}")
            st.write("Prediction Probabilities:")
            for i, prob in enumerate(predictions):
                st.write(f"{classes[i]}: {prob:.2%}")

if __name__ == '__main__':
    main()

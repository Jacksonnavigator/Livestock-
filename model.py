import tensorflow as tf

def load_model():
    # Load a pre-trained model (e.g., a simple model trained on livestock images)
    model = tf.keras.models.load_model('path/to/your/model')
    return model

def predict(model, img_array):
    predictions = model.predict(img_array)
    # Assuming binary classification for simplicity
    if predictions[0] > 0.5:
        return "Sick"
    else:
        return "Healthy"

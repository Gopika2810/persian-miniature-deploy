import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Persian Miniature Art Classifier", layout="centered")

st.title("🎨 Persian Miniature Art Classifier")
st.write("Upload an image to classify its artistic style or theme.")

# --- Model and Class Configuration ---

MODEL_PATH = "persian_miniature_model.keras"
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1DOtZhvuYxGgy9SZCgcTqbFROJvp68p1M"

# --- Class Names from your Training Notebook ---
CLASS_NAMES = [
    'Herat',
    'Qajar',
    'Shiraz-e Avval',
    'Tabriz-e Avval',
    'Tabriz-e Dovvom'
] #

# --- Model Loading ---

# Download the model from Google Drive if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading the model from Google Drive... This may take a moment."):
        try:
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.error("Please ensure the Google Drive link is public and accessible ('Anyone with the link').")
            st.stop()

# Load the TensorFlow model (cached for performance)
@st.cache_resource
def load_keras_model():
    """Loads the Keras model from the specified path."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        st.stop()

model = load_keras_model()

# --- Image Upload and Prediction ---

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.write("")
        st.write("Classifying...")

        # Preprocess the image for the model
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)

        # Confidence threshold to detect out-of-domain images
        CONFIDENCE_THRESHOLD = 0.50 # You can adjust this value

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ **Low Confidence:** The model is not very confident about the prediction ({confidence*100:.2f}%).")
            st.info("This image may not be a Persian Miniature painting from the categories the model was trained on.")
        else:
            predicted_class_name = CLASS_NAMES[predicted_index]
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

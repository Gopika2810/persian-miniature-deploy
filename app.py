import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Title
st.title("🎨 Persian Miniature Classifier")
st.write("Upload an image and the model will classify it into a Persian miniature type.")

# Mapping of class indices to painting names
class_names = [
    "Royal Court Scene",
    "Battle Scene",
    "Nature Scene",
    "Religious Scene",
    "Mythological Scene"
]

# Load model with caching
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("persian_miniature_model.keras")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img_resized = img.resize((224, 224))  # adjust if your model input size is different
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)

        # Confidence threshold to handle random/out-of-domain images
        threshold = 0.5

        if confidence < threshold:
            st.warning("⚠ The uploaded image may not belong to any known Persian miniature class.")
        else:
            predicted_name = class_names[predicted_index]
            st.success(f"Prediction: {predicted_name} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f" Error processing image: {e}")

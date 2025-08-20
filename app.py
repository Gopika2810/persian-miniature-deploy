import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

st.title(" Persian Miniature Classifier")
st.write("Upload an image and the model will classify it into Persian Miniature art classes.")

# ----------------------
# Download model from Google Drive if not exists
# ----------------------
MODEL_PATH = "persian_miniature_model.keras"
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1DOtZhvuYxGgy9SZCgcTqbFROJvp68p1M"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait"):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# ----------------------
# Load model
# ----------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ----------------------
# Upload image
# ----------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    try:
        img = Image.open(uploaded_file).convert("RGB").resize((224, 224))  # adjust size if different
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class_index]

        # You can replace these with your actual class names
        class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

        # Threshold for out-of-domain images
        if confidence < 0.5:
            st.warning(" The image may not belong to the Persian Miniature domain. Try another image.")
        else:
            st.success(f"Prediction: {class_names[predicted_class_index]} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

st.title(" Persian Miniature Classifier")
st.write("Upload an image and the model will classify it.")

MODEL_PATH = "persian_miniature_model.keras"

# Google Drive direct download link
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1DOtZhvuYxGgy9SZCgcTqbFROJvp68p1M"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... this may take a minute"):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.write(f"### Prediction: Class {predicted_class}")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("persian_miniature_model.keras")
    return model

model = load_model()

st.title(" Persian Miniature Classifier")
st.write("Upload an image and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # adjust input size if different
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.write(f"###  Prediction: Class {predicted_class}")

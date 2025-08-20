import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("persian_miniature_model.keras")
    return model

model = load_model()

# Class names in your dataset
class_names = [
    "Court Scene",
    "Battle Scene",
    "Religious Scene",
    "Landscape",
    "Portrait",
    "Animal Scene"
]

st.title("Persian Miniature Classifier")
st.write("Upload an image and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    try:
        img = image.load_img(uploaded_file, target_size=(224, 224))  # adjust size if needed
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        # Threshold for out-of-domain images
        if confidence < 0.5:
            st.warning("The uploaded image may not belong to Persian Miniature paintings.")
        else:
            st.success(f"Prediction: {class_names[predicted_class_index]} (Confidence: {confidence:.2f})")
    except Exception as e:
        st.error("Error processing image. Make sure it's a valid image file.")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("outputs/best_model.keras")

model = load_model()

st.title("Cats vs Dogs Image Classifier")
st.write("Upload an image to predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = image.resize((160, 160))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    prediction = model.predict(img_array, verbose=0)[0][0]

    label = "Dog" if prediction >= 0.5 else "Cat"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

# -*- coding: utf-8 -*-
"""streamlit_app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Qu7na6q88HYXuGPe0TqyMhDuOKmo8bgW
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('try.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit interface
st.title("Pendeteksi Lubang pada Jalan Berbasis CNN")

uploaded_file = st.file_uploader("Unggah gambar jalan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    prediction = predict(image)
    class_names = ["Jalan Normal", "Jalan Berlubang"]
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Prediksi: {predicted_class}")

    feedback = st.radio("Apakah prediksi ini tepat?", ("Ya", "Tidak"))
    if feedback:
        st.write("Terima kasih atas feedback Anda!")

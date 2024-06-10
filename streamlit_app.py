import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
import tensorflow as tf

# Fungsi untuk memuat model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "fourlayer.h5"  # Ganti dengan nama model Anda
    model = tf.keras.models.load_model(model_path)
    return model

# Muat model
model = load_model()

# Fungsi untuk memproses gambar yang diunggah
def preprocess_image(image):
    img = image.resize((224, 224))  # Sesuaikan ukuran gambar sesuai input model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

# Fungsi untuk memproses dan melakukan prediksi pada beberapa gambar
def predict(images):
    processed_images = np.vstack([preprocess_image(image) for image in images])
    predictions = model.predict(processed_images)
    return predictions

# Fungsi untuk memuat gambar dari URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    return img

# Antarmuka Streamlit
st.title("Pothole Detection System Using Convolutional Neural Network")

# Pilihan untuk mengunggah gambar atau memasukkan URL
option = st.selectbox("Pilih metode input gambar:", ("Unggah gambar", "Masukkan URL gambar"))

if option == "Unggah gambar":
    uploaded_files = st.file_uploader("Unggah gambar jalan", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]

        for image in images:
            st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        if st.button("Klasifikasi Gambar"):
            predictions = predict(images)
            class_names = ["Jalan berlubang", "Jalan normal"]

            for i, prediction in enumerate(predictions):
                predicted_class = class_names[np.argmax(prediction)]
                st.write(f"Prediksi untuk {uploaded_files[i].name}: {predicted_class}")

            st.write("Apakah prediksi ini tepat?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Ya"):
                    st.write("Terima kasih atas feedback Anda!")

            with col2:
                if st.button("Tidak"):
                    st.write("Terima kasih, kami akan memperbaiki model kami berdasarkan feedback Anda!")

elif option == "Masukkan URL gambar":
    url = st.text_input("Masukkan URL gambar jalan:")

    if url:
        try:
            image = load_image_from_url(url)
            st.image(image, caption='Gambar dari URL', use_column_width=True)

            predictions = predict([image])
            class_names = ["Jalan berlubang", "Jalan normal"]
            predicted_class = class_names[np.argmax(predictions[0])]
            st.write(f"Prediksi: {predicted_class}")

            st.write("Apakah prediksi ini tepat?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Ya"):
                    st.write("Terima kasih atas feedback Anda!")

            with col2:
                if st.button("Tidak"):
                    st.write("Terima kasih, kami akan memperbaiki model kami berdasarkan feedback Anda!")

        except Exception as e:
            st.error(f"Gagal memuat gambar dari URL: {e}")

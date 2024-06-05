import streamlit as st
from PIL import Image
import numpy as np
import requests
import io
import tensorflow as tf

# Fungsi untuk memuat model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = "second/TDG.h5"
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

# Fungsi untuk melakukan prediksi
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

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
    uploaded_file = st.file_uploader("Unggah gambar jalan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        prediction = predict(image)
        class_names = ["Jalan berlubang", "Jalan normal"]
        predicted_class = class_names[np.argmax(prediction)]
        st.write(f"Prediksi: {predicted_class}")

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

            prediction = predict(image)
            class_names = ["Jalan berlubang", "Jalan normal"]
            predicted_class = class_names[np.argmax(prediction)]
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

import streamlit as st
# UBAH BAGIAN INI: Impor Interpreter langsung dari tflite_runtime
import tensorflow as tf 
import numpy as np
from PIL import Image

# Judul aplikasi
st.title("WEB DETEKSI PENYAKIT DAUN TANAMAN UBI JALAR")
st.write("By Indra Herdiana TI-4B")

# Load model TFLite
@st.cache_resource
def load_model():
    # UBAH BAGIAN INI: Panggil Interpreter secara langsung, bukan melalui tf.lite
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Ambil detail input & output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kelas yang sesuai dengan model kamu
class_names = ["bercak bulat hitam", "bercak hitam tepi kuning", "sehat"]

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img = image.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Hasil prediksi
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.markdown(f"### Prediksi: **{predicted_class}**")
    st.write(f"Tingkat Akurasi: {confidence:.2f}%")
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Judul aplikasi
st.title("WEB DETEKSI PENYAKIT DAUN TANAMAN UBI JALAR")
st.write("By Indra Herdiana TI-4B")

# Load model TFLite
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_VGG16.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Ambil detail input & output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["bercak bulat hitam", "bercak hitam tepi kuning", "sehat"]

# BARU: Dictionary untuk menyimpan tips perawatan
tips_perawatan = {
    "sehat": """
    ### Tips Merawat Daun Agar Tetap Sehat
    Daun tanaman Anda dalam kondisi prima! Berikut cara untuk menjaganya:
    - **Penyiraman Teratur:** Siram tanaman secukupnya saat tanah mulai kering. Hindari menyiram daun secara langsung, fokus pada area akar untuk mengurangi risiko jamur.
    - **Pemupukan Seimbang:** Berikan pupuk kompos atau pupuk NPK seimbang secara berkala untuk memastikan nutrisi tanaman tercukupi.
    - **Sinar Matahari Cukup:** Pastikan tanaman mendapatkan sinar matahari penuh, setidaknya 6-8 jam setiap hari.
    - **Inspeksi Rutin:** Periksa daun secara berkala (misalnya seminggu sekali) untuk mendeteksi gejala penyakit atau hama sejak dini.
    - **Jaga Kebersihan Area Tanam:** Bersihkan gulma atau sisa tanaman yang membusuk di sekitar tanaman Anda.
    """,
    "bercak bulat hitam": """
    ### Cara Mengatasi Penyakit Bercak Bulat Hitam (Cercospora)
    Penyakit ini umumnya disebabkan oleh jamur dan sering menyebar di kondisi lembap. Berikut langkah-langkah penanganannya:
    - **Sanitasi & Pemangkasan:** Segera pangkas dan musnahkan (bakar) daun yang terinfeksi untuk mencegah penyebaran spora jamur. Jangan buang di tumpukan kompos.
    - **Perbaiki Sirkulasi Udara:** Atur jarak tanam atau pangkas sebagian daun yang terlalu rimbun agar sirkulasi udara lebih lancar dan daun lebih cepat kering setelah hujan atau penyiraman.
    - **Hindari Penyiraman Berlebih:** Siram tanaman di pagi hari pada bagian akarnya, bukan daunnya.
    - **Penggunaan Fungisida (Jika Perlu):** Jika serangan sudah parah, pertimbangkan untuk menggunakan fungisida nabati atau kimia yang mengandung bahan aktif seperti *mankozeb* atau tembaga. Selalu ikuti petunjuk pada label kemasan.
    - **Rotasi Tanaman:** Hindari menanam ubi jalar di lokasi yang sama setiap musim untuk memutus siklus hidup jamur di tanah.
    """,
    "bercak hitam tepi kuning": """
    ### Cara Mengatasi Bercak Hitam Tepi Kuning
    Gejala ini bisa disebabkan oleh beberapa faktor, termasuk infeksi jamur, bakteri, atau kekurangan nutrisi. Berikut penanganan awalnya:
    - **Isolasi & Pemangkasan:** Sama seperti penyakit bercak daun, segera potong dan buang daun yang menunjukkan gejala ini untuk menghentikan penyebaran.
    - **Periksa Kondisi Tanah:** Pastikan drainase tanah baik dan tidak tergenang air. Akar yang terendam air rentan terhadap penyakit.
    - **Cek Nutrisi:** Tepi daun yang menguning bisa menjadi tanda kekurangan kalium. Pertimbangkan untuk memberikan pupuk yang kaya akan Kalium (K).
    - **Gunakan Pestisida Nabati:** Semprotkan pestisida nabati seperti ekstrak bawang putih atau daun mimba yang dapat bertindak sebagai anti-jamur dan anti-bakteri alami.
    """
}

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
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

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    threshold = 97.0

    st.write("---") # Garis pemisah
    
    # DIUBAH: Menambahkan logika untuk menampilkan tips
    if confidence >= threshold:
        if predicted_class == "sehat":
            st.success(f"Prediksi: Daun SEHAT (Keyakinan: {confidence:.2f}%)")
        else:
            st.error(f"Prediksi: Terdeteksi {predicted_class.upper()} (Keyakinan: {confidence:.2f}%)")
        
        # Tampilkan tips perawatan yang sesuai
        st.markdown("---")
        tips = tips_perawatan[predicted_class]
        st.markdown(tips)

    else:
        st.warning(f"Gambar tidak dapat diklasifikasikan dengan yakin.")
        st.write(f"Tingkat keyakinan model hanya **{confidence:.2f}%**, yang berada di bawah ambang batas **{threshold}%**.")
        st.info("Pastikan gambar yang Anda unggah adalah foto daun ubi jalar yang jelas.")
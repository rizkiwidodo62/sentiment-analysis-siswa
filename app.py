import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# Pastikan library yang digunakan di model juga ada di requirements.txt

# Muat Model dan Vectorizer yang sudah disimpan
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model atau Vectorizer tidak ditemukan. Jalankan model.py terlebih dahulu.")
    st.stop()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Mapping hasil sentimen
sentimen_map = {
    0: 'Negatif 🔴',
    1: 'Positif 🟢',
    2: 'Netral 🟡'
}

def preprocess_text(text):
    # Contoh preprocessing sederhana
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Hapus non-alphanumeric
    return text

def predict_sentiment(text):
    # 1. Preprocess
    clean_text = preprocess_text(text)
    # 2. Vectorize
    vectorized_text = vectorizer.transform([clean_text])
    # 3. Predict
    prediction = model.predict(vectorized_text)[0]
    return sentimen_map[prediction]

# --- Antarmuka Streamlit ---
st.set_page_config(page_title="Analisis Sentimen Peserta Didik", layout="centered")

st.title("💡 Analisis Sentimen Umpan Balik Peserta Didik")
st.markdown("Masukkan teks umpan balik dari peserta didik di bawah ini untuk menganalisis sentimennya (Positif, Negatif, atau Netral).")

# Area input teks
user_input = st.text_area("Teks Umpan Balik:", height=150, placeholder="Contoh: Saya suka cara guru mengajar, materinya mudah dipahami.")

# Tombol untuk analisis
if st.button("Analisis Sentimen"):
    if user_input:
        with st.spinner('Menganalisis...'):
            result = predict_sentiment(user_input)
            st.success("✅ Analisis Selesai!")
            st.markdown(f"### Hasil Sentimen: **{result}**")
            st.balloons()
    else:
        st.warning("Mohon masukkan teks umpan balik untuk dianalisis.")
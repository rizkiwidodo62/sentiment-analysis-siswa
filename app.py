import streamlit as st
import pickle
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator # Import BaseEstimator untuk type hinting

# Mapping hasil sentimen (sesuaikan dengan model Anda)
SENTIMENT_MAP = {
    0: 'Negatif 🔴',
    1: 'Positif 🟢',
    2: 'Netral 🟡'
}

st.set_page_config(page_title="Upload & Analisis Model", layout="wide")

st.title("⬆️ Upload Model & Data untuk Analisis Sentimen")
st.markdown("Unggah Model Sentimen (`.pkl`) dan Data Peserta Didik (`.xlsx`/`.csv`) untuk melakukan analisis secara massal.")

# --- Bagian Upload File ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Unggah Model Sentimen (Pickle)")
    uploaded_model_file = st.file_uploader(
        "Pilih file .pkl yang berisi model sentimen Anda.", 
        type=['pkl']
    )

with col2:
    st.subheader("2. Unggah Data Peserta Didik (Excel/CSV)")
    uploaded_data_file = st.file_uploader(
        "Pilih file .xlsx atau .csv untuk dianalisis.", 
        type=['xlsx', 'csv']
    )

# --- Bagian Proses dan Analisis ---
if uploaded_model_file is not None and uploaded_data_file is not None:
    st.markdown("---")
    st.subheader("3. Proses Analisis")
    
    # Memuat Model dan Vectorizer
    try:
        # File .pkl harus berisi TUPLE: (model, vectorizer) atau class yang berisi keduanya.
        # Jika Anda menyimpan model dan vectorizer terpisah, gabungkan di sini.
        uploaded_model_data = pickle.load(uploaded_model_file)
        
        # Asumsi: file .pkl berisi (model, vectorizer)
        if isinstance(uploaded_model_data, tuple) and len(uploaded_model_data) == 2:
            model = uploaded_model_data[0]
            vectorizer = uploaded_model_data[1]
        else:
            # Jika file pkl berisi objek yang lebih kompleks atau model tunggal, sesuaikan di sini.
            st.error("Format file .pkl tidak sesuai. Harap pastikan berisi (model, vectorizer) atau struktur yang dikenali.")
            st.stop()

        st.success("✅ Model dan Vectorizer berhasil dimuat.")
        
    except Exception as e:
        st.error(f"Gagal memuat file model (.pkl): {e}")
        st.stop()
        
    # Memuat Data
    try:
        if uploaded_data_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_data_file)
        elif uploaded_data_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_data_file)
        else:
            st.warning("Jenis file data tidak didukung.")
            st.stop()
        
        st.success("✅ Data berhasil dimuat.")
        st.dataframe(data.head())

    except Exception as e:
        st.error(f"Gagal memuat file data: {e}")
        st.stop()

    # Logika Analisis
    
    # Minta pengguna menentukan kolom teks
    st.markdown("---")
    text_column = st.selectbox(
        "Pilih nama **kolom** di data yang berisi teks umpan balik:", 
        options=data.columns
    )
    
    if st.button("Jalankan Analisis Sentimen Massal"):
        if text_column in data.columns:
            with st.spinner('Menganalisis sentimen untuk semua data...'):
                
                # Fungsi prediksi (Pastikan data di-preprocess seperti saat training)
                def predict_sentiment_from_model(text):
                    if pd.isna(text) or text == "":
                        return SENTIMENT_MAP.get(2, 'Netral') # Anggap missing/kosong sebagai Netral
                    try:
                        vectorized_text = vectorizer.transform([str(text).lower()])
                        prediction = model.predict(vectorized_text)[0]
                        return SENTIMENT_MAP.get(prediction, 'Tidak Diketahui')
                    except Exception as e:
                        return f"Error: {e}"

                # Terapkan fungsi prediksi ke kolom teks
                data['Hasil_Sentimen'] = data[text_column].apply(predict_sentiment_from_model)
                
                st.success("Analisis Sentimen Massal Selesai!")
                
                # Tampilkan hasil
                st.dataframe(data)

                # Sediakan tombol unduh hasil
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Unduh Hasil Analisis (.csv)",
                    data=csv,
                    file_name='hasil_analisis_sentimen.csv',
                    mime='text/csv',
                )
        else:
            st.error("Kolom yang dipilih tidak ditemukan.")

elif uploaded_model_file is None and uploaded_data_file is None:
    st.info("Silakan unggah Model dan Data Anda di atas untuk memulai.")
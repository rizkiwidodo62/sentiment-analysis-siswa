import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# --- 1. Simulasikan Data Pelatihan ---
data = {
    'teks': [
        "Pembelajaran hari ini sangat menyenangkan.",
        "Materi yang disampaikan terlalu sulit dan membingungkan.",
        "Guru menjelaskannya dengan sangat baik dan detail.",
        "Tugasnya terlalu banyak, saya jadi stres.",
        "Saya tidak punya komentar tentang materi ini.",
        "Metode pengajaran ini efisien sekali.",
        "Lingkungan kelas kurang kondusif.",
        "Saya merasa termotivasi setelah sesi ini.",
        "Waktu yang diberikan terlalu singkat."
    ],
    # 1: Positif, 0: Negatif, 2: Netral
    'sentimen': [1, 0, 1, 0, 2, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# --- 2. Feature Extraction (Vectorizer) ---
# TfidfVectorizer harus di-fit pada data training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['teks'])
y = df['sentimen']

# --- 3. Train Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# --- 4. Gabungkan dan Simpan sebagai Tuple ---

# Menggabungkan model (classifier) dan vectorizer menjadi satu tuple
# Urutan: (model, vectorizer)
model_package = (model, vectorizer)

# Tentukan nama file output
output_filename = 'sentiment_package.pkl'

# Menyimpan tuple ke dalam file .pkl
try:
    with open(output_filename, 'wb') as f:
        pickle.dump(model_package, f)
    
    print("----------------------------------------------------------------")
    print(f"✅ Model dan Vectorizer berhasil disimpan sebagai satu file:")
    print(f"   Nama File: {output_filename}")
    print(f"   Ukuran File: {os.path.getsize(output_filename) / 1024:.2f} KB")
    print("----------------------------------------------------------------")

except Exception as e:
    print(f"❌ Gagal menyimpan file: {e}")
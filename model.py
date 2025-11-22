import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# --- Simulasikan Data ---
data = {
    'teks': [
        "Pembelajaran hari ini sangat menyenangkan.",
        "Materi yang disampaikan terlalu sulit.",
        "Guru menjelaskannya dengan baik.",
        "Tugasnya terlalu banyak, saya jadi stres.",
        "Saya tidak punya komentar."
    ],
    # 1: Positif, 0: Negatif, 2: Netral
    'sentimen': [1, 0, 1, 0, 2]
}
df = pd.DataFrame(data)
# ------------------------

# 1. Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['teks'])
y = df['sentimen']

# 2. Train Model
model = LogisticRegression()
model.fit(X, y)

# 3. Simpan Model dan Vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model dan Vectorizer berhasil disimpan.")
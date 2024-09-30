import streamlit as st
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# Konfigurasi halaman
st.set_page_config(page_title="Sentiment Analysis - Positive/Negative", layout="wide")

# Sidebar untuk informasi aplikasi
with st.sidebar:
    st.title("🔍 Sentiment Classifier")
    st.markdown("""
        ### Steps to classify sentiment:
        1. Upload your dataset.
        2. Select text and label columns.
        3. Train the SVM model.
        4. See the results for positive and negative sentiment!
    """)
    st.info("💡 This app allows you to analyze text sentiment as positive or negative using a Support Vector Machine (SVM) model.")

# Bagian utama aplikasi
st.title("🔍 Positive/Negative Sentiment Analysis with SVM")
st.markdown("""
    Welcome to the **Sentiment Analysis App**! This app classifies text into **positive** or **negative** sentiment using an SVM classifier.
""")

# Unggah file CSV untuk analisis sentimen
uploaded_file = st.file_uploader("📂 Upload your sentiment dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Membaca file CSV
    data = pd.read_csv(uploaded_file)
    st.write("📊 Data yang diunggah:")
    st.dataframe(data.head())

    # Menampilkan kolom data untuk memverifikasi nama kolom
    st.write("Kolom yang tersedia dalam dataset:", data.columns)

    # Pemilihan nama kolom yang benar untuk fitur dan label
    feature_column = st.selectbox("Pilih kolom teks (Fitur)", data.columns)
    label_column = st.selectbox("Pilih kolom label (Sentimen)", data.columns)

    # Tombol untuk memproses data dan melatih model
    if st.button("🚀 Process Data and Train Model"):
        # Gunakan nama kolom yang dipilih dari dropdown
        X = data[feature_column]
        y = data[label_column]

        # Split data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ekstraksi fitur menggunakan CountVectorizer (konversi teks menjadi angka)
        vectorizer = CountVectorizer()
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)

        # Melatih model SVM
        clf = svm.SVC(kernel='linear', C=1.0, max_iter=-1)  # Menggunakan kernel linear
        clf.fit(X_train_vect, y_train)

        # Memprediksi hasil
        y_pred = clf.predict(X_test_vect)

        # Evaluasi model
        accuracy = clf.score(X_test_vect, y_test)
        classification_report = metrics.classification_report(y_test, y_pred)

        # Menampilkan hasil evaluasi
        st.subheader("🎯 Model Accuracy")
        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

        st.subheader("📋 Classification Report")
        st.text(classification_report)

        # Menampilkan Confusion Matrix
        st.subheader("🔄 Confusion Matrix")
        cm = metrics.confusion_matrix(y_test, y_pred)
        st.write(cm)

else:
    st.warning("Please upload a CSV file to start!")

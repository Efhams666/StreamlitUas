# =====================================================
#           TEMPLATE FILE: classification.py
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================
# Fungsi Load Data
# =====================================================
def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)

# =====================================================
# Preprocessing
# =====================================================
def preprocess(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y

# =====================================================
# Training Model
# =====================================================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return model, acc, report, matrix

# =====================================================
# Streamlit App
# =====================================================
st.title("📊 Sistem Klasifikasi Machine Learning")
st.write("Upload dataset → train model → lihat hasil klasifikasi.")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df)

    try:
        X, y = preprocess(df)

        if st.button("Train Model"):
            model, acc, report, matrix = train_model(X, y)

            st.success(f"Model berhasil dilatih! Akurasi: {acc:.2f}")

            st.write("### Classification Report")
            st.text(report)

            st.write("### Confusion Matrix")
            st.write(matrix)

    except Exception as e:
        st.error(f"Error: {e}")

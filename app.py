
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

# Load model
model = joblib.load('model.pkl')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Risiko Kanker Paru-paru",
    page_icon="🫁",
)
st.title("Dashboard Prediksi Risiko Kanker Paru-paru")
st.write("Masukkan data responden untuk memprediksi kemungkinan mengidap kanker paru-paru.")

st.markdown("---")
st.subheader("Evaluasi Model")

# Load dataset
data = pd.read_csv('survey_lung_cancer_updated.csv')

# Tampilkan unique value kolom untuk pengecekan
st.write("Unique value GENDER:", data['GENDER'].unique())
st.write("Unique value ANXIETY:", data['ANXIETY'].unique())

# Persiapan data
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Prediksi
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Evaluasi metrik
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, y_prob)

# Tampilkan metrik evaluasi
col1, col2, col3, col4 = st.columns(4)
col1.success(f"Accuracy: **{acc:.2f}**")
col2.info(f"Precision: **{prec:.2f}**")
col3.warning(f"Recall: **{rec:.2f}**")
col4.error(f"ROC AUC: **{roc_auc:.2f}**")

# Grafik ROC dan Confusion Matrix
plot_option = st.selectbox("Pilih grafik untuk ditampilkan:", ["Pilih", "ROC AUC Curve", "Confusion Matrix"])
if plot_option == "ROC AUC Curve":
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    st.pyplot(fig2)

elif plot_option == "Confusion Matrix":
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

st.markdown("---")
st.subheader("Analisis Rumusan Masalah")

# Analisis Risiko berdasarkan Gender
risk_by_gender = data.groupby('GENDER')['LUNG_CANCER'].mean().reset_index()
male_data = risk_by_gender[risk_by_gender['GENDER'] == 'MALE']
female_data = risk_by_gender[risk_by_gender['GENDER'] == 'FEMALE']

if not male_data.empty:
    st.write(f"Rata-rata risiko kanker paru-paru laki-laki: {male_data['LUNG_CANCER'].values[0]:.2f}")
else:
    st.write("Data laki-laki tidak ditemukan.")

if not female_data.empty:
    st.write(f"Rata-rata risiko kanker paru-paru perempuan: {female_data['LUNG_CANCER'].values[0]:.2f}")
else:
    st.write("Data perempuan tidak ditemukan.")

st.markdown("-----")

# Analisis Risiko berdasarkan Anxiety
risk_by_anxiety = data.groupby('ANXIETY')['LUNG_CANCER'].mean().reset_index()
anxiety_yes = risk_by_anxiety[risk_by_anxiety['ANXIETY'] == 'YES']
anxiety_no = risk_by_anxiety[risk_by_anxiety['ANXIETY'] == 'NO']

if not anxiety_yes.empty:
    st.write(f"Rata-rata risiko kanker paru-paru untuk yang memiliki anxiety (YES): {anxiety_yes['LUNG_CANCER'].values[0]:.2f}")
else:
    st.write("Data anxiety YES tidak ditemukan.")

if not anxiety_no.empty:
    st.write(f"Rata-rata risiko kanker paru-paru untuk yang tidak memiliki anxiety (NO): {anxiety_no['LUNG_CANCER'].values[0]:.2f}")
else:
    st.write("Data anxiety NO tidak ditemukan.")

st.markdown("---")
st.subheader("Prediksi Risiko Individu")

with st.form("prediction_form"):
    GENDER = st.selectbox("Jenis Kelamin", ["MALE", "FEMALE"])
    AGE = st.slider("Usia", 15, 100, 40)
    SMOKING = st.selectbox("Apakah merokok?", ["YES", "NO"])
    YELLOW_FINGERS = st.selectbox("Jari Menguning?", ["YES", "NO"])
    ANXIETY = st.selectbox("Kecemasan?", ["YES", "NO"])
    PEER_PRESSURE = st.selectbox("Tekanan dari teman sebaya?", ["YES", "NO"])
    CHRONIC_DISEASE = st.selectbox("Penyakit Kronis?", ["YES", "NO"])
    FATIGUE = st.selectbox("Sering lelah?", ["YES", "NO"])
    ALLERGY = st.selectbox("Alergi?", ["YES", "NO"])
    WHEEZING = st.selectbox("Bengek?", ["YES", "NO"])
    ALCOHOL = st.selectbox("Konsumsi alkohol?", ["YES", "NO"])
    COUGHING = st.selectbox("Sering batuk?", ["YES", "NO"])
    SHORTNESS_OF_BREATH = st.selectbox("Sesak napas?", ["YES", "NO"])
    SWALLOWING_DIFFICULTY = st.selectbox("Sulit menelan?", ["YES", "NO"])
    CHEST_PAIN = st.selectbox("Nyeri dada?", ["YES", "NO"])

    submit = st.form_submit_button("Prediksi")

# Mapping input ke bentuk numerik
def map_inputs():
    binary_map = {"YES": 1, "NO": 0, "MALE": 1, "FEMALE": 0}
    return pd.DataFrame([{
        'GENDER': binary_map[GENDER],
        'AGE': AGE,
        'SMOKING': binary_map[SMOKING],
        'YELLOW_FINGERS': binary_map[YELLOW_FINGERS],
        'ANXIETY': binary_map[ANXIETY],
        'PEER_PRESSURE': binary_map[PEER_PRESSURE],
        'CHRONIC DISEASE': binary_map[CHRONIC_DISEASE],
        'FATIGUE ': binary_map[FATIGUE],
        'ALLERGY ': binary_map[ALLERGY],
        'WHEEZING': binary_map[WHEEZING],
        'ALCOHOL CONSUMING': binary_map[ALCOHOL],
        'COUGHING': binary_map[COUGHING],
        'SHORTNESS OF BREATH': binary_map[SHORTNESS_OF_BREATH],
        'SWALLOWING DIFFICULTY': binary_map[SWALLOWING_DIFFICULTY],
        'CHEST PAIN': binary_map[CHEST_PAIN]
    }])

if submit:
    input_df = map_inputs()
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"Hasil prediksi: **{'Berpotensi Mengidap Kanker Paru-paru' if pred == 1 else 'Tidak Terindikasi Kanker Paru-paru'}**")
    st.write(f"Probabilitas kanker: **{prob:.2f}**")

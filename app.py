
import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# ----------------------------------
# Load & Preprocessing
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("survey_lung_cancer.csv")
    df = df.dropna().drop_duplicates()
    df.columns = df.columns.str.strip()  # Bersihkan spasi

    if df['LUNG_CANCER'].dtype == object:
        df['LUNG_CANCER'] = df['LUNG_CANCER'].str.upper()
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df = df.applymap(lambda x: 1 if x == 2 else x)
    return df

df = load_data()
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']

# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Ringkasan", "Clustering", "Supervised Learning"])

# ----------------------------------
# RINGKASAN
# ----------------------------------
if menu == "Ringkasan":
    st.title("Ringkasan Dataset")
    st.write(df.head())
    st.write("Jumlah data:", df.shape[0])
    st.write("Jumlah fitur:", df.shape[1])
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

# ----------------------------------
# CLUSTERING
# ----------------------------------
elif menu == "Clustering":
    st.title("Clustering KMeans (n=3)")

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['CLUSTER'] = kmeans.fit_predict(scaled_X)

    st.subheader("Distribusi Cluster terhadap LUNG_CANCER")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='CLUSTER', hue='LUNG_CANCER', data=df, palette='Set2', ax=ax1)
    st.pyplot(fig1)

    st.subheader("Distribusi Usia per Cluster")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='CLUSTER', y='AGE', data=df, palette='pastel', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Heatmap Korelasi")
    corr_data = df.select_dtypes(include='number').drop(columns=['LUNG_CANCER'])
    corr_data = corr_data.loc[:, corr_data.nunique() > 1]  # drop fitur konstan
    corr_matrix = corr_data.corr()
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

# ----------------------------------
# SUPERVISED LEARNING
# ----------------------------------
elif menu == "Supervised Learning":
    st.title("Model Supervised Learning")

    model_choice = st.selectbox("Model Sekarang", ["Naive Bayes"])  # Bisa ditambah Logistic Regression

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Akurasi Model")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi: {acc:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ===============================
    # INPUT MANUAL + PREDIKSI
    # ===============================
    st.subheader("Prediksi Baru (Manual Input)")

    input_data = {
        'AGE': st.slider("Usia", int(df['AGE'].min()), int(df['AGE'].max()), int(df['AGE'].mean())),
        'GENDER': st.radio("Jenis Kelamin", options={1: "Laki-laki", 0: "Perempuan"}, format_func=lambda x: {1: "Laki-laki", 0: "Perempuan"}[x]),
        'SMOKING': st.radio("Merokok?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'YELLOW_FINGERS': st.radio("Jari Kuning?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'ANXIETY': st.radio("Kecemasan?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'PEER_PRESSURE': st.radio("Tekanan Teman Sebaya?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'CHRONIC DISEASE': st.radio("Penyakit Kronis?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'FATIGUE': st.radio("Sering Lelah?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'ALLERGY': st.radio("Alergi?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'WHEEZING': st.radio("Sesak Napas (Wheezing)?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'ALCOHOL CONSUMING': st.radio("Konsumsi Alkohol?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'COUGHING': st.radio("Batuk Kronis?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'SHORTNESS OF BREATH': st.radio("Sesak Napas?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'SWALLOWING DIFFICULTY': st.radio("Sulit Menelan?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
        'CHEST PAIN': st.radio("Nyeri Dada?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak"),
    }

    if st.button("Predict"):
        sample = pd.DataFrame([input_data])[X.columns]  # cocokkan urutan kolom
        pred = model.predict(sample)[0]
        st.success(f"Hasil Prediksi: {'YES' if pred == 1 else 'NO'}")

        probs = model.predict_proba(sample)[0]
        st.info(f"Peluang Tidak Kanker: {probs[0]*100:.2f}%")
        st.info(f"Peluang Kanker: {probs[1]*100:.2f}%")

        st.subheader("Rata-rata Fitur untuk Kelas")
        means_df = pd.DataFrame(model.theta_, columns=X.columns, index=['NO', 'YES'])
        st.dataframe(means_df)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load model
model = joblib.load('model.pkl')

# ----------------------------------
# Load & Preprocessing
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("survey_lung_cancer.csv")
    df = df.dropna().drop_duplicates()

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
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.drop(columns=['LUNG_CANCER']).corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# ----------------------------------
# SUPERVISED LEARNING
# ----------------------------------
elif menu == "Supervised Learning":
    st.title("Model Supervised Learning")

    # Pilih model
    model_choice = st.selectbox("Model Sekarang", ["Naive Bayes"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inisialisasi dan latih model
    #if model_choice == "Logistic Regression":
        #model = LogisticRegression(max_iter=1000)
    #else:
    model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Akurasi Model")
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi: {acc:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ===============================
    # INPUT MANUAL + TOMBOL PREDIKSI
    # ===============================
    st.subheader("Prediksi Baru (Manual Input)")
    input_data = {}
    for col in X.columns:
        if df[col].nunique() <= 2:
            input_data[col] = st.selectbox(f"{col}", [0, 1])
        else:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            input_data[col] = st.slider(f"{col}", min_val, max_val, int(df[col].mean()))

    # Tombol Prediksi
    if st.button("Predict"):
        sample = pd.DataFrame([input_data])
        pred = model.predict(sample)[0]
        st.success(f"Hasil Prediksi: {'YES' if pred == 1 else 'NO'}")

        if model_choice == "Naive Bayes":
            probs = model.predict_proba(sample)[0]
            st.info(f"Peluang Tidak Kanker: {probs[0]*100:.2f}%")
            st.info(f"Peluang Kanker: {probs[1]*100:.2f}%")

            st.subheader("Rata-rata Fitur untuk Kelas")
            means_df = pd.DataFrame(model.theta_, columns=X.columns, index=['NO', 'YES'])
            st.dataframe(means_df)

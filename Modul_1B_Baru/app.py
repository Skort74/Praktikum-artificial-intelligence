import streamlit as st
import pandas as pd
import joblib
import json
import os

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Gagal Jantung (DNN)", page_icon="❤️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Muat Model & Info
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "model_heart.pkl"))

model = load_model()

with open(os.path.join(BASE_DIR, "model_info.json")) as f:
    info = json.load(f)

# 3. Sidebar Info
with st.sidebar:
    st.header("Informasi Model")
    st.write(f"**Model:** {info['nama_model']}")
    st.metric("Accuracy", f"{info['metrik']['accuracy']*100:.1f}%")
    st.metric("F1 Macro", f"{info['metrik']['f1_macro']:.3f}")
    st.caption(f"Dievaluasi pada {info['metrik']['n_test']} data uji.")

st.title("❤️ Klasifikasi Risiko Kematian Pasien Gagal Jantung (DNN)")
st.write("Masukkan data klinis pasien untuk memprediksi status keselamatan pasien.")

# 4. Tab Navigasi
tab1, tab2 = st.tabs(["🔮 Prediksi Satu Data", "📁 Prediksi dari CSV"])

# --- TAB 1: Form Input ---
with tab1:
    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Usia", 20, 100, 60)
            anaemia = st.selectbox("Anaemia (0=Tidak, 1=Ya)", [0, 1])
            cpk = st.number_input("Creatinine Phosphokinase (CPK)", 0, 10000, 500)
            diabetes = st.selectbox("Diabetes (0=Tidak, 1=Ya)", [0, 1])
            
        with col2:
            ef = st.number_input("Ejection Fraction (%)", 10, 80, 38)
            hbp = st.selectbox("Hipertensi (0=Tidak, 1=Ya)", [0, 1])
            platelets = st.number_input("Platelets", 0.0, 1000000.0, 250000.0)
            creatinine = st.number_input("Serum Creatinine", 0.0, 10.0, 1.0)
            
        with col3:
            sodium = st.number_input("Serum Sodium", 100, 150, 137)
            sex = st.selectbox("Jenis Kelamin (1=Laki-laki, 0=Perempuan)", [0, 1])
            smoking = st.selectbox("Merokok (0=Tidak, 1=Ya)", [0, 1])
            time_obs = st.number_input("Waktu Observasi (hari)", 0, 300, 100)

        submitted = st.form_submit_button("Prediksi Sekarang")
        
        if submitted:
            data_input = {
                "age": age, "anaemia": anaemia, "creatinine_phosphokinase": cpk,
                "diabetes": diabetes, "ejection_fraction": ef, "high_blood_pressure": hbp,
                "platelets": platelets, "serum_creatinine": creatinine, "serum_sodium": sodium,
                "sex": sex, "smoking": smoking, "time": time_obs
            }
            
            df_input = pd.DataFrame([data_input])
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            
            st.divider()
            st.subheader("Hasil Prediksi")
            
            if pred == 1:
                st.error("### ⚠️ Pasien Diprediksi **MENINGGAL (1)**")
            else:
                st.success("### ✅ Pasien Diprediksi **SELAMAT (0)**")
                
            st.write("**Tingkat Keyakinan Model:**")
            st.progress(float(proba[1]), text=f"Probabilitas Meninggal: {proba[1]*100:.2f}%")
            st.progress(float(proba[0]), text=f"Probabilitas Selamat: {proba[0]*100:.2f}%")

# --- TAB 2: Prediksi CSV ---
with tab2:
    st.subheader("Unggah File CSV")
    st.write("Unggah file CSV dengan 12 kolom fitur klinis (tanpa kolom target `DEATH_EVENT`).")
    
    # Fitur unduhan sampel (Sesuai permintaan tugas no.5)
    if os.path.exists(os.path.join(BASE_DIR, "dataset_heart.csv")):
        with st.expander("Belum punya file CSV? Unduh sampel di sini."):
            sample_df = pd.read_csv(os.path.join(BASE_DIR, "dataset_heart.csv"))
            sample_df = sample_df.drop(columns=["DEATH_EVENT", "bagian"]).head(10)
            st.download_button("📥 Unduh Sampel CSV", sample_df.to_csv(index=False), "sampel_jantung.csv", "text/csv")

    file = st.file_uploader("Pilih file CSV", type=["csv"])
    if file:
        df_upload = pd.read_csv(file)
        
        required_cols = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", 
                         "ejection_fraction", "high_blood_pressure", "platelets", 
                         "serum_creatinine", "serum_sodium", "sex", "smoking", "time"]
        
        if st.button("Prediksi Semua Baris"):
            if all(col in df_upload.columns for col in required_cols):
                preds = model.predict(df_upload[required_cols])
                probas = model.predict_proba(df_upload[required_cols])[:, 1]
                
                df_upload["Prediksi_DNN"] = ["Meninggal" if p == 1 else "Selamat" for p in preds]
                df_upload["Probabilitas_Meninggal (%)"] = (probas * 100).round(2)
                
                st.dataframe(df_upload)
                st.download_button("📥 Unduh Hasil Prediksi", df_upload.to_csv(index=False), "hasil_prediksi_jantung.csv", "text/csv")
            else:
                st.error("Kolom pada file CSV tidak lengkap.")
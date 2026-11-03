import streamlit as st
import pandas as pd
import time
import os
from groq import Groq
from classifier import NAMA_FITUR, classify, load_config

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Gagal Jantung (LLM)", page_icon="🤖", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Muat Konfigurasi dari Modul 2A
cfg = load_config(os.path.join(BASE_DIR, "config.json"))
FITUR = list(NAMA_FITUR.keys())

# 3. Inisialisasi Groq Client
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) atau menu Secrets (Streamlit Cloud).")
    st.stop()

@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)

client = get_client()

# 4. Fungsi Prediksi dengan Cache (Hemat Kuota)
@st.cache_data(show_spinner=False)
def prediksi(age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking, time_obs):
    data = {
        "age": age, "anaemia": anaemia, "creatinine_phosphokinase": cpk,
        "diabetes": diabetes, "ejection_fraction": ef, "high_blood_pressure": hbp,
        "platelets": platelets, "serum_creatinine": creatinine, "serum_sodium": sodium,
        "sex": sex, "smoking": smoking, "time": time_obs
    }
    return classify(client, cfg, data)

# 5. Sidebar Info
ev = cfg["evaluasi"]
with st.sidebar:
    st.header("Tentang Model LLM")
    st.write(f"**Model:** `{cfg['model']}`")
    st.write(f"**Few-shot:** {len(cfg['few_shot'])} contoh")
    st.metric("F1 Macro (Data Uji)", f"{ev['f1_macro']:.3f}")
    st.metric("Accuracy (Data Uji)", f"{ev['accuracy']*100:.1f}%")
    st.caption(f"Dievaluasi pada {ev['n']} data uji, tanggal {ev['tanggal']}")

st.title("🤖 Klasifikasi Risiko Gagal Jantung (Groq LLM)")
st.write("Masukkan data klinis pasien. Model AI akan memprediksi apakah pasien diprediksi **Selamat (0)** atau **Meninggal (1)**.")

# 6. Tab Navigasi
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Satu Data", "📁 Prediksi dari CSV", "📥 Unduh Sampel Data"])

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

        submitted = st.form_submit_button("Prediksi dengan AI")
        
        if submitted:
            with st.spinner("Menghubungi Groq LLM..."):
                hasil = prediksi(age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking, time_obs)
                
                st.divider()
                st.subheader("Hasil Prediksi LLM")
                
                if hasil == "1":
                    st.error("### ⚠️ Pasien Diprediksi **MENINGGAL (1)**")
                elif hasil == "0":
                    st.success("### ✅ Pasien Diprediksi **SELAMAT (0)**")
                else:
                    st.warning("Model tidak dapat memberikan jawaban yang valid. Silakan coba lagi.")

# --- TAB 2: Prediksi CSV ---
with tab2:
    st.subheader("Unggah File CSV")
    st.write("Unggah CSV dengan 12 kolom klinis. (Maksimal 30 baris agar hemat kuota API).")
    
    file = st.file_uploader("Pilih file CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30) # Batasi 30 baris
            st.dataframe(df.head()) # Tampilkan preview
            
            if st.button("Prediksi Semua dengan LLM"):
                bar = st.progress(0.0)
                hasil = []
                
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    # Panggil fungsi prediksi
                    pred = prediksi(
                        int(b["age"]), int(b["anaemia"]), int(b["creatinine_phosphokinase"]),
                        int(b["diabetes"]), int(b["ejection_fraction"]), int(b["high_blood_pressure"]),
                        float(b["platelets"]), float(b["serum_creatinine"]), int(b["serum_sodium"]),
                        int(b["sex"]), int(b["smoking"]), int(b["time"])
                    )
                    hasil.append("Meninggal" if pred == "1" else "Selamat" if pred == "0" else "Invalid")
                    bar.progress(i / len(df))
                    time.sleep(3) # Jeda 3 detik per request (Wajib)
                    
                df["Prediksi_LLM"] = hasil
                st.dataframe(df)
                st.download_button("📥 Unduh Hasil", df.to_csv(index=False), "hasil_prediksi_llm.csv", "text/csv")

# --- TAB 3: Unduh Sampel Data ---
with tab3:
    st.subheader("Unduh Sampel Data Uji")
    st.write("Gunakan file ini untuk mencoba fitur prediksi massal di tab kedua.")
    if os.path.exists(os.path.join(BASE_DIR, "dataset_heart.csv")):
        sample_df = pd.read_csv(os.path.join(BASE_DIR, "dataset_heart.csv"))
        sample_df = sample_df.drop(columns=["DEATH_EVENT", "bagian"]).head(10)
        st.dataframe(sample_df)
        st.download_button(
            label="📥 Unduh Sampel CSV",
            data=sample_df.to_csv(index=False).encode('utf-8'),
            file_name='sampel_jantung.csv',
            mime='text/csv',
        )
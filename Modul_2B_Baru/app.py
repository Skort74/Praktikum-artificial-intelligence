import streamlit as st
import pandas as pd
import time
import os
import traceback
from groq import Groq
from classifier import NAMA_FITUR, classify, load_config

st.set_page_config(page_title="Prediksi Gagal Jantung (LLM)", page_icon="🤖", layout="wide")

# KITA BUNGKUS SEMUA INISIALISASI DALAM TRY-EXCEPT
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Cek file config.json
    if not os.path.exists(os.path.join(BASE_DIR, "config.json")):
        raise FileNotFoundError("File config.json tidak ditemukan di folder ini!")
        
    cfg = load_config(os.path.join(BASE_DIR, "config.json"))
    FITUR = list(NAMA_FITUR.keys())
    
    # 2. Cek API Key
    try:
        API_KEY = st.secrets["GROQ_API_KEY"]
        if not API_KEY.startswith("gsk_"):
            raise ValueError("Format API Key salah! Harus diawali dengan 'gsk_'")
    except KeyError:
        raise KeyError("GROQ_API_KEY tidak ditemukan di menu Secrets Streamlit Cloud!")
        
    # 3. Inisialisasi Groq
    @st.cache_resource
    def get_client():
        return Groq(api_key=API_KEY)
    client = get_client()

except Exception as e:
    st.error("### 🚨 APLIKASI CRASH SAAT INISIALISASI!")
    st.write("Silakan **copy-paste** teks di dalam kotak abu-abu di bawah ini ke chat saya:")
    st.code(traceback.format_exc(), language="python")
    st.stop() # Hentikan aplikasi agar tidak loading terus

# ================= KODE UI DI BAWAH INI SAMA SEPERTI SEBELUMNYA =================

ev = cfg.get("evaluasi", {})
with st.sidebar:
    st.header("Tentang Model LLM")
    st.write(f"**Model:** `{cfg.get('model', 'Tidak diketahui')}`")
    st.write(f"**Few-shot:** {len(cfg.get('few_shot', []))} contoh")
    if ev:
        st.metric("F1 Macro (Data Uji)", f"{ev.get('f1_macro', 0):.3f}")
        st.metric("Accuracy (Data Uji)", f"{ev.get('accuracy', 0)*100:.1f}%")

st.title("🤖 Klasifikasi Risiko Gagal Jantung (Groq LLM)")
st.write("Masukkan data klinis pasien.")

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Satu Data", "📁 Prediksi dari CSV", "📥 Unduh Sampel Data"])

@st.cache_data(show_spinner=False)
def prediksi(age, anaemia, cpk, diabetes, ef, hbp, platelets, creatinine, sodium, sex, smoking, time_obs):
    data = {
        "age": age, "anaemia": anaemia, "creatinine_phosphokinase": cpk,
        "diabetes": diabetes, "ejection_fraction": ef, "high_blood_pressure": hbp,
        "platelets": platelets, "serum_creatinine": creatinine, "serum_sodium": sodium,
        "sex": sex, "smoking": smoking, "time": time_obs
    }
    return classify(client, cfg, data)

with tab1:
    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Usia", 20, 100, 60)
            anaemia = st.selectbox("Anaemia (0=Tidak, 1=Ya)", [0, 1])
            cpk = st.number_input("CPK", 0, 10000, 500)
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
                if hasil == "1":
                    st.error("### ⚠️ Pasien Diprediksi **MENINGGAL (1)**")
                elif hasil == "0":
                    st.success("### ✅ Pasien Diprediksi **SELAMAT (0)**")
                else:
                    st.warning("Model tidak dapat memberikan jawaban yang valid. Silakan coba lagi.")

with tab2:
    st.subheader("Unggah File CSV")
    file = st.file_uploader("Pilih file CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            st.dataframe(df.head())
            if st.button("Prediksi Semua dengan LLM"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    pred = prediksi(
                        int(b["age"]), int(b["anaemia"]), int(b["creatinine_phosphokinase"]),
                        int(b["diabetes"]), int(b["ejection_fraction"]), int(b["high_blood_pressure"]),
                        float(b["platelets"]), float(b["serum_creatinine"]), int(b["serum_sodium"]),
                        int(b["sex"]), int(b["smoking"]), int(b["time"])
                    )
                    hasil.append("Meninggal" if pred == "1" else "Selamat" if pred == "0" else "Invalid")
                    bar.progress(i / len(df))
                    time.sleep(3)
                df["Prediksi_LLM"] = hasil
                st.dataframe(df)
                st.download_button("📥 Unduh Hasil", df.to_csv(index=False), "hasil_prediksi_llm.csv", "text/csv")

with tab3:
    st.subheader("Unduh Sampel Data Uji")
    dataset_path = os.path.join(BASE_DIR, "dataset_heart.csv")
    if os.path.exists(dataset_path):
        sample_df = pd.read_csv(dataset_path).drop(columns=["DEATH_EVENT", "bagian"]).head(10)
        st.dataframe(sample_df)
        st.download_button("📥 Unduh Sampel CSV", sample_df.to_csv(index=False).encode('utf-8'), "sampel_jantung.csv", "text/csv")
    else:
        st.warning("File 'dataset_heart.csv' tidak ditemukan di repositori Anda.")
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Prediksi Status Akademik",
    page_icon="🎓",
    layout="centered",
)

# Path absolut ke folder app.py — aman dipakai di lokal maupun Streamlit Cloud
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "model_status_akademik.pkl")

@st.cache_data
def load_info():
    with open(BASE_DIR / "model_info.json", "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
info = load_info()

FITUR = info["fitur_numerik"] + info["fitur_kategori"]
RENTANG = info.get("rentang_fitur", {})
METRIK = info.get("metrik", {})
VERSI = info.get("versi", {})

# --- 2. Sidebar: informasi model ---
with st.sidebar:
    st.header("Tentang Model")
    st.write(info.get("nama_model", "DNN"))

    if "accuracy" in METRIK:
        st.metric("Accuracy (data uji)", f"{METRIK['accuracy']:.1%}")
    if "f1_macro" in METRIK:
        st.metric(
            "F1 macro (data uji)",
            f"{METRIK['f1_macro']:.3f}",
            delta=f"baseline {METRIK.get('f1_baseline', 0):.2f}"
        )
    st.caption(
        f"Dievaluasi pada {METRIK.get('n_test', '-')} data uji. "
        f"scikit-learn {VERSI.get('scikit-learn', '-')}"
    )
    st.caption("Kelas: " + ", ".join(info.get("kelas", [])))

st.title("🎓 Prediksi Status Akademik Mahasiswa")
st.write(
    "Masukkan data mahasiswa untuk memprediksi status: "
    "**Aman**, **Waspada**, atau **Berisiko**."
)

tab1, tab2 = st.tabs(["Prediksi Satu Mahasiswa", "Prediksi dari File CSV"])

# --- 3. Prediksi satu data ---
with tab1:
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            ipk = st.number_input(
                "IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01
            )
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
            jam_belajar = st.slider("Jam belajar mandiri per minggu", 0, 40, 8)
        with col2:
            tugas = st.slider("Tugas terkumpul (%)", 0, 100, 80)
            uts = st.slider("Nilai UTS", 0, 100, 70)
            organisasi = st.radio(
                "Aktif organisasi?", ["Ya", "Tidak"], horizontal=True
            )
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        data = pd.DataFrame([{
            "ipk": ipk,
            "kehadiran": kehadiran,
            "jam_belajar": jam_belajar,
            "tugas_terkumpul": tugas,
            "nilai_uts": uts,
            "aktif_organisasi": organisasi,
        }])[FITUR]

        # Cek rentang data latih
        luar = []
        for c, batas in RENTANG.items():
            if c in data.columns:
                nilai = data[c].iloc[0]
                if not (batas[0] <= nilai <= batas[1]):
                    luar.append(f"{c} ({batas[0]}–{batas[1]})")
        if luar:
            st.warning(
                "Nilai di luar rentang data latih: "
                + ", ".join(luar)
                + ". Model kurang dapat diandalkan di luar rentang ini."
            )

        pred = model.predict(data)[0]
        st.success(f"Prediksi status: **{pred}**")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(data)[0]
            prob_df = pd.DataFrame({
                "Status": model.classes_,
                "Probabilitas": proba,
            }).set_index("Status")
            st.bar_chart(prob_df)
            st.caption(f"Keyakinan model: {proba.max():.1%}")

# --- 4. Prediksi dari CSV ---
with tab2:
    st.write(
        "Unggah file CSV dengan kolom: "
        + ", ".join(f"`{c}`" for c in FITUR)
    )
    file = st.file_uploader("Pilih file CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]

        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            hasil = df.copy()
            hasil["prediksi_status"] = model.predict(hasil[FITUR])

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(hasil[FITUR])
                for i, kelas in enumerate(model.classes_):
                    hasil[f"prob_{kelas}"] = proba[:, i]

            st.dataframe(hasil)
            st.write(hasil["prediksi_status"].value_counts())

            st.download_button(
                "Unduh hasil prediksi",
                data=hasil.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv",
                mime="text/csv",
            )
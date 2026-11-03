import json, os, time, pandas as pd
from datetime import date
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, f1_score, classification_report
from classifier import NAMA_FITUR, classify

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

df = pd.read_csv("dataset_heart.csv")
FITUR = list(NAMA_FITUR)
LABELS = ["0", "1"]

train = df[df["bagian"] == "train"]
test = df[df["bagian"] == "test"]

# 1. Ambil 10 data validasi per kelas
val = train.groupby("DEATH_EVENT").sample(10, random_state=7)
val["status"] = val["DEATH_EVENT"].astype(str)
pool = train.drop(val.index)

def evaluasi(cfg, data):
    prediksi = []
    for i, (_, baris) in enumerate(data.iterrows(), start=1):
        prediksi.append(classify(client, cfg, baris[FITUR].to_dict()))
        print(f"\rProses: {i}/{len(data)}", end="", flush=True)
        time.sleep(3) # Jeda 3 detik
    print()
    y_true = data["status"].tolist()
    return {
        "accuracy": round(accuracy_score(y_true, prediksi), 4),
        "f1_macro": round(f1_score(y_true, prediksi, labels=LABELS, average="macro", zero_division=0), 4),
        "invalid": prediksi.count("invalid"), "n": len(data)
    }

PROMPT_DASAR = (
    "Kamu adalah sistem prediksi kematian pasien gagal jantung.\n"
    "Data: Usia, Anaemia, CPK, Diabetes, Ejection Fraction, Hipertensi, Platelets, Kreatinin, Sodium, Sex, Merokok, Waktu.\n"
    "ATURAN KETAT: Jawab HANYA dengan satu karakter angka. "
    "Tulis 0 jika pasien diprediksi selamat, atau tulis 1 jika pasien diprediksi meninggal. "
    "Jangan berikan penjelasan apapun."
)

# --- Iterasi 1: Zero-shot ---
# MENGGUNAKAN MODEL QWEN YANG TERBUKTI BERHASIL
cfg_v1 = {"model": "qwen/qwen3.8-27b", "temperature": 0, "max_tokens": 128, "labels": LABELS, "system_prompt": PROMPT_DASAR, "few_shot": []}
print("--- Iterasi 1 (Zero-shot) ---")
hasil_v1 = evaluasi(cfg_v1, val)
print(hasil_v1)

# --- Iterasi 2: Few-shot ---
contoh = pool.groupby("DEATH_EVENT").sample(2, random_state=1)
contoh["status"] = contoh["DEATH_EVENT"].astype(str)
cfg_v2 = dict(cfg_v1)
cfg_v2["few_shot"] = [{"data": b[FITUR].to_dict(), "label": b["status"]} for _, b in contoh.iterrows()]

print("\n--- Iterasi 2 (Few-shot) ---")
hasil_v2 = evaluasi(cfg_v2, val)
print(hasil_v2)

# --- Pilih Terbaik & Uji di Test Set ---
kandidat = [(hasil_v1, cfg_v1), (hasil_v2, cfg_v2)]
hasil_val, cfg_final = max(kandidat, key=lambda k: k[0]["f1_macro"])
print(f"\nKonfigurasi terpilih: Iterasi dengan F1 validasi {hasil_val['f1_macro']}")

# Uji pada 30 data test
test_sample = test.sample(n=min(30, len(test)), random_state=42)
test_sample["status"] = test_sample["DEATH_EVENT"].astype(str)
print(f"\n--- Evaluasi Test ({len(test_sample)} data) ---")
hasil_test = evaluasi(cfg_final, test_sample)
print(hasil_test)

# --- Bandingkan dengan DNN ---
if os.path.exists("model_info.json"):
    with open("model_info.json") as f:
        dnn = json.load(f)["metrik"]
    print("\n=== Perbandingan DNN vs LLM ===")
    print(pd.DataFrame({
        "DNN (Modul 1A)": [dnn["accuracy"], dnn["f1_macro"]],
        "LLM (Modul 2A)": [hasil_test["accuracy"], hasil_test["f1_macro"]],
    }, index=["accuracy", "f1_macro"]))
    hasil_test["f1_dnn_pembanding"] = dnn["f1_macro"]

# --- Gerbang Evaluasi & Simpan config.json ---
TARGET_F1 = 0.70 # Target standar
if hasil_test["f1_macro"] >= TARGET_F1:
    cfg_final["evaluasi"] = {**hasil_test, "tanggal": date.today().isoformat()}
    with open("config.json", "w") as f:
        json.dump(cfg_final, f, indent=2)
    print("\n✅ LOLOS evaluasi -> config.json siap dipakai Streamlit (Modul 2B)")
else:
    print(f"\n❌ BELUM LOLOS (F1 {hasil_test['f1_macro']} < {TARGET_F1})")
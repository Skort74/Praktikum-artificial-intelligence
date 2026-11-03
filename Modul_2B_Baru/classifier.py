import json
import re
import time
import groq

# Daftar fitur dataset Jantung
NAMA_FITUR = {
    "age": "Usia", "anaemia": "Anaemia (0/1)", "creatinine_phosphokinase": "CPK",
    "diabetes": "Diabetes (0/1)", "ejection_fraction": "Ejection Fraction",
    "high_blood_pressure": "Hipertensi (0/1)", "platelets": "Platelets",
    "serum_creatinine": "Kreatinin", "serum_sodium": "Sodium", "sex": "Laki-laki (1/0)",
    "smoking": "Merokok (0/1)", "time": "Waktu Observasi"
}

def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def format_data(data):
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())

def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs

def parse_label(output, labels):
    # Fungsi pintar untuk membaca 0 atau 1
    if re.search(r'\b1\b', output): return "1"
    if re.search(r'\b0\b', output): return "0"
    return "invalid"

def classify(client, cfg, data):
    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            temperature=0,
            max_tokens=128, # Beri token cukup agar model tidak terpotong
            messages=build_messages(cfg, data),
        )
        return parse_label(resp.choices[0].message.content, cfg["labels"])
    except Exception as e:
        print(f"\n⚠️ API Error: {e}")
        time.sleep(5)
        return "invalid"
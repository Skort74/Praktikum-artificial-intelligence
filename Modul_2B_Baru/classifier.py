
import json
import re
import groq


# ============================================================
# 1. DAFTAR FITUR DATASET JANTUNG
# ============================================================

NAMA_FITUR = {
    "age": "Usia",
    "anaemia": "Anaemia (0/1)",
    "creatinine_phosphokinase": "CPK",
    "diabetes": "Diabetes (0/1)",
    "ejection_fraction": "Ejection Fraction",
    "high_blood_pressure": "Hipertensi (0/1)",
    "platelets": "Platelets",
    "serum_creatinine": "Kreatinin",
    "serum_sodium": "Sodium",
    "sex": "Laki-laki (1/0)",
    "smoking": "Merokok (0/1)",
    "time": "Waktu Observasi"
}


# ============================================================
# 2. MEMUAT KONFIGURASI MODEL
# ============================================================

def load_config(path="config.json"):
    """
    Memuat konfigurasi model dari file JSON.
    """

    try:
        with open(path, "r", encoding="utf-8") as file:
            cfg = json.load(file)

        required_keys = [
            "model",
            "system_prompt",
            "few_shot",
            "labels",
            "evaluasi"
        ]

        for key in required_keys:
            if key not in cfg:
                raise ValueError(
                    f"Konfigurasi '{key}' tidak ditemukan di config.json"
                )

        return cfg

    except FileNotFoundError:
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {path}"
        )

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Format JSON tidak valid: {e}"
        )


# ============================================================
# 3. FORMAT DATA INPUT
# ============================================================

def format_data(data):
    """
    Mengubah data pasien menjadi format teks
    berdasarkan urutan fitur dataset.
    """

    missing_features = [
        feature for feature in NAMA_FITUR
        if feature not in data
    ]

    if missing_features:
        raise ValueError(
            f"Fitur data tidak lengkap: {missing_features}"
        )

    return "; ".join(
        f"{nama}: {data[kolom]}"
        for kolom, nama in NAMA_FITUR.items()
    )


# ============================================================
# 4. MEMBANGUN PESAN FEW-SHOT PROMPTING
# ============================================================

def build_messages(cfg, data):
    """
    Menyusun pesan system, few-shot examples,
    dan data pasien yang akan diklasifikasikan.
    """

    messages = [
        {
            "role": "system",
            "content": cfg["system_prompt"]
        }
    ]

    # Memasukkan contoh data dan label sebagai few-shot
    for example in cfg.get("few_shot", []):

        if "data" not in example or "label" not in example:
            raise ValueError(
                "Format few-shot tidak valid. "
                "Setiap contoh harus memiliki data dan label."
            )

        messages.append({
            "role": "user",
            "content": format_data(example["data"])
        })

        messages.append({
            "role": "assistant",
            "content": str(example["label"])
        })

    # Memasukkan data pasien yang akan diprediksi
    messages.append({
        "role": "user",
        "content": format_data(data)
    })

    return messages


# ============================================================
# 5. PARSING DAN VALIDASI LABEL
# ============================================================

def parse_label(output, labels):
    """
    Mengambil label klasifikasi dari respons LLM.

    Hanya menerima:
    - Jawaban langsung berupa 0 atau 1.
    - Format eksplisit seperti Prediksi: 0 atau Label: 1.

    Respons yang ambigu akan dianggap invalid.
    """

    if not output or not isinstance(output, str):
        return "invalid"

    valid_labels = [str(label) for label in labels]

    # Format jawaban langsung: 0 atau 1
    match = re.fullmatch(
        r'\s*["\']?([01])["\']?\s*[.!]?\s*',
        output
    )

    if match:
        label = match.group(1)

        if label in valid_labels:
            return label

    # Format jawaban eksplisit:
    # Prediksi: 0
    # Label: 1
    # Klasifikasi: 0
    match = re.search(
        r'^\s*(?:prediksi|label|klasifikasi|jawaban)'
        r'\s*[:=]\s*["\']?([01])["\']?\s*[.!]?\s*$',
        output,
        re.IGNORECASE
    )

    if match:
        label = match.group(1)

        if label in valid_labels:
            return label

    return "invalid"


# ============================================================
# 6. FUNGSI KLASIFIKASI MENGGUNAKAN GROQ LLM
# ============================================================

def classify(client, cfg, data):
    """
    Melakukan klasifikasi risiko gagal jantung
    menggunakan model Large Language Model melalui Groq API.

    Return:
    - "0"       : Pasien diprediksi selamat.
    - "1"       : Pasien diprediksi meninggal.
    - "invalid" : Respons tidak valid atau terjadi error.
    """

    try:

        # Validasi konfigurasi
        if not cfg.get("model"):
            raise ValueError("Nama model tidak ditemukan.")

        # Membangun pesan untuk LLM
        messages = build_messages(cfg, data)

        # Mengirim permintaan ke Groq API
        response = client.chat.completions.create(
            model=cfg["model"],
            temperature=0,
            max_tokens=128,
            messages=messages
        )

        # Mengambil respons model
        output = response.choices[0].message.content

        # Memvalidasi hasil klasifikasi
        hasil = parse_label(
            output,
            cfg["labels"]
        )

        if hasil == "invalid":
            print(
                f"Respons LLM tidak sesuai format: {output}"
            )

        return hasil

    except groq.APIError as e:

        print(f"Groq API Error: {e}")
        return "invalid"

    except (ValueError, KeyError, TypeError) as e:

        print(f"Data atau konfigurasi tidak valid: {e}")
        return "invalid"

    except Exception as e:

        print(f"Classification Error: {e}")
        return "invalid"

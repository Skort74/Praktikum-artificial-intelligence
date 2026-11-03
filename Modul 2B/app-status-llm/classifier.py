"""Klasifikasi status akademik melalui Groq untuk evaluasi dan deploy.

Model, prompt, contoh few-shot, dan parameter dibaca dari konfigurasi yang
sama agar perilaku saat evaluasi (Modul 2A) dan deploy (Modul 2B) konsisten.
"""

import json
import re
import time

import groq


NAMA_FITUR = {
    "ipk": "IPK",
    "kehadiran": "Kehadiran (%)",
    "jam_belajar": "Jam belajar mandiri per minggu",
    "tugas_terkumpul": "Tugas terkumpul (%)",
    "nilai_uts": "Nilai UTS",
    "aktif_organisasi": "Aktif organisasi",
}


def load_config(path="config.json"):
    """Baca konfigurasi yang telah lolos evaluasi."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def format_data(data):
    """Ubah dict data seorang mahasiswa menjadi satu baris teks."""
    return "; ".join(
        f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items()
    )


def build_messages(cfg, data):
    """Susun instruksi, contoh berlabel, lalu data yang akan diprediksi."""
    messages = [{"role": "system", "content": cfg["system_prompt"]}]
    for example in cfg.get("few_shot", []):
        messages.append({"role": "user", "content": format_data(example["data"])})
        messages.append({"role": "assistant", "content": example["label"]})
    messages.append({"role": "user", "content": format_data(data)})
    return messages


def parse_label(output, labels):
    """Rapikan label; jawaban kosong atau di luar daftar dihitung invalid."""
    answer = re.sub(r"[^a-z]", "", (output or "").lower())
    label_map = {label.lower(): label for label in labels}
    return label_map.get(answer, "invalid")


def classify(client, cfg, data, max_retry=3):
    """Panggil model sesuai konfigurasi, dengan retry saat rate limit."""
    for attempt in range(max_retry):
        try:
            options = {}
            if "reasoning_effort" in cfg:
                options["reasoning_effort"] = cfg["reasoning_effort"]
            response = client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
                **options,
            )
            return parse_label(response.choices[0].message.content, cfg["labels"])
        except groq.RateLimitError:
            time.sleep(15 * (attempt + 1))
    return "invalid"

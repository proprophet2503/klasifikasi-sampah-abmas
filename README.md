---
title: Klasifikasi Sampah
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Klasifikasi Sampah

Aplikasi web untuk mengklasifikasikan foto sampah ke dalam lima kategori,
dibangun dari [template deployment](../template/README.md) di repositori ini.

Blok di paling atas file ini adalah metadata Hugging Face Spaces. Biarkan apa
adanya jika aplikasi akan di-deploy ke Spaces; abaikan saja jika hanya dipakai lokal.

## Model

| Aspek | Nilai |
|---|---|
| Sumber | Teachable Machine (MobileNet) |
| File | `models/keras_model.h5` (2,3 MB) |
| Input | 224 × 224 × 3, normalisasi `(x/127.5) - 1` |
| Output | 5 kelas, sudah softmax |

Kelas pada `models/labels.txt` dan nama tampilannya:

| Label model | Ditampilkan sebagai |
|---|---|
| `Organic_Full` | Organik |
| `B3_Full` | B3 (Bahan Berbahaya & Beracun) |
| `Cardboard Primer` | Kertas & Kardus |
| `Recycle Primer` | Daur Ulang |
| `Residu Primer` | Residu |

Pemetaan nama diatur di `config.yaml` bagian `label_display`, sehingga
`labels.txt` tidak perlu diubah.

## Menjalankan secara lokal

```bash
cd klasifikasi-sampah-app

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Buka `http://localhost:8501`, lalu unggah foto sampah.

## Menyesuaikan tampilan

Semua penyesuaian dilakukan di `config.yaml`:

- `app.title`, `app.subtitle`, `app.about` — teks yang tampil
- `inference.confidence_threshold` — ambang batas peringatan "kurang yakin" (kini 0.60)
- `inference.top_k` — jumlah kandidat teratas yang ditampilkan (kini 3)
- `label_display` — nama kelas versi Bahasa Indonesia

Referensi lengkap tiap kunci ada di [template/README.md](../template/README.md#4-referensi-lengkap-configyaml).

## Deploy ke Hugging Face Spaces

Langkah lengkap ada di
[template/README.md bagian 7](../template/README.md#7-deploy-ke-hugging-face-spaces).
Ringkasnya: buat Space dengan SDK **Streamlit**, unggah seluruh isi folder ini
(termasuk `core/`, `models/`, dan `.streamlit/`) dengan struktur folder yang
sama, lalu tunggu build selesai.

# Sistem Diagnosis Penyakit Daun Anggur

Aplikasi web berbasis **Streamlit** untuk mendiagnosis penyakit daun anggur menggunakan tiga model deep learning PyTorch:

1. **Segmentasi daun** — U-Net + EfficientNet-B0
2. **Segmentasi lesi** — U-Net + EfficientNet-B0
3. **Klasifikasi penyakit** — EfficientNet-B0 (3 kelas)

## Struktur Project

```
Website/
├── app.py                          # Aplikasi Streamlit
├── requirements.txt
├── README.md
├── models/
│   ├── model_segmentasi_daun_efficientnetb0_best.pth
│   ├── model_segmentasi_lesi_efficientnetb0_best.pth
│   └── klasifikasi_penyakit_final_best.pth
├── notebooks/                      # Notebook training (referensi)
└── utils/
    ├── model_loader.py             # Arsitektur & load checkpoint
    ├── preprocessing.py            # Letterbox, normalisasi, post-processing
    ├── inference.py                # Pipeline inference
    └── severity.py                 # Perhitungan severity & kategori
```

## Pipeline Sistem

1. User mengupload citra daun anggur
2. Citra di-resize letterbox ke 256×256 (padding putih)
3. Model segmentasi daun → `leaf mask` (threshold 0,45)
4. Background di-blackout → citra masked
5. Model segmentasi lesi → `lesion mask` (threshold 0,40), dibatasi area daun
6. Severity = (piksel lesi / piksel daun) × 100%
7. Jika severity ≤ 0,5% → **Sehat**
8. Jika severity > 0,5% → model klasifikasi (input 224×224, citra masked)

### Kategori Severity

| Kategori | Rentang |
|----------|---------|
| Sehat    | 0% – 0,5% |
| Ringan   | >0,5% – 25% |
| Sedang   | >25% – 50% |
| Berat    | >50% |

### Kelas Klasifikasi

- Black Measles
- Black Rot
- Isariopsis Leaf Spot

## Instalasi

```bash
# Buat virtual environment
python -m venv venv

# Aktivasi (Windows)
venv\Scripts\activate

# Aktivasi (Linux/macOS)
source venv/bin/activate

# Install dependency
pip install -r requirements.txt
```

## Menjalankan Aplikasi

**Cara termudah (Windows):** double-click `run.bat` di folder project.

```bash
# Aktivasi venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Aktivasi venv (Windows CMD)
venv\Scripts\activate.bat

# Jalankan
streamlit run app.py
```

Aplikasi akan terbuka di browser (default: `http://localhost:8501`).

## Catatan Teknis

- **Device:** CUDA otomatis jika tersedia, fallback ke CPU
- **Preprocessing segmentasi:** ImageNet normalize via Albumentations, letterbox 256×256
- **Preprocessing klasifikasi:** Resize 224×224, ImageNet normalize via torchvision
- Arsitektur model mengikuti notebook di folder `notebooks/` — tidak diubah

# Sign Language Recognition (LSTM + MediaPipe)

Real-time **Sign Language Recognition** menggunakan **MediaPipe Hands** dan **LSTM Neural Network** untuk mengenali gesture statis & dinamis berbasis urutan frame.

---

## 🎯 Fitur Utama

* Realtime hand tracking (2 tangan)
* Gesture statis & dinamis
* Audio feedback (Text-to-Speech via `.wav`)
* Temporal smoothing untuk mengurangi salah deteksi
* Modular & siap dikembangkan

---

## 📂 Struktur Project

```
signlag_ml/
├── scripts/
│   ├── collect_data.py        # Ambil dataset gesture
│   ├── prepare_dataset.py     # Preprocessing & split data
│   ├── train_lstm.py          # Training model LSTM
│   └── realtime_predict.py    # Realtime prediction
├── sounds/                    # Audio output gesture
├── tests/                     # Testing utilitas
├── data/                      # (DI-IGNORE) dataset npy
├── models/                    # (DI-IGNORE) model .h5
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Konsep Model

* Input: sequence `(30, 126)` → 30 frame, 2 tangan (21 landmark × 3 koordinat)
* Model: LSTM → Dense Softmax
* Output: gesture label

Gesture dinamis dibedakan melalui **perubahan temporal antar frame**, bukan pose tunggal.

---

## 🎥 Cara Menjalankan

### 1️⃣ Install Dependency

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Realtime Detection

```bash
python scripts/realtime_predict.py
```

---

## 🔊 Audio Output

Setiap gesture memiliki file audio `.wav` yang akan diputar **saat confidence stabil**.

---

## 🛠 Solusi Salah Deteksi Gesture

### Masalah Umum:

* Gesture mirip (`N` vs `nama`)
* Gesture lambat terdeteksi

### Solusi yang Diterapkan:

1. **Confidence Gap Filter**

   * Gesture diterima hanya jika selisih confidence Top-1 & Top-2 > 0.2
2. **Temporal Consistency**

   * Gesture harus stabil selama beberapa frame
3. **Cooldown Audio**

   * Mencegah spam audio

---

## 📌 Catatan Penting

* Dataset & model **tidak dipublikasikan** (private)
* Model dapat dilatih ulang dengan gesture tambahan

---

## 🚀 Pengembangan Selanjutnya

* Left / Right hand separation
* Transformer-based sequence model
* Mobile deployment

---

## 👤 Author

**Nuril Aisyahroni**
Machine Learning & Computer Vision Enthusiast

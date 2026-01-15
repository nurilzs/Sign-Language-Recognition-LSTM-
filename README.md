# Sign Language Recognition (LSTM)

Proyek pengenalan bahasa isyarat menggunakan **MediaPipe Hands** dan **LSTM (TensorFlow)**.
Aplikasi ini dapat mengenali gesture statis dan dinamis secara real-time melalui webcam
dan mengeluarkan **audio suara** sesuai gesture yang terdeteksi.

## Gesture yang Didukung
- halo
- nama
- saya
- Salam
- I
- L
- N
- R
- U

## Teknologi
- Python 3.9
- OpenCV
- MediaPipe
- TensorFlow (LSTM)
- NumPy
- Scikit-learn
- Playsound

## Struktur Folder
signlag_ml/
├── data/
│ └── raw/
├── models/
│ └── sign_language_lstm.h5
├── sounds/
├── scripts/
│ ├── collect_data.py
│ ├── prepare_dataset.py
│ ├── train_lstm.py
│ └── realtime_predict.py
├── requirements.txt
└── README.md


## Cara Menjalankan
```bash
pip install -r requirements.txt
python scripts/realtime_predict.py


## note guys
- gesture dynamic membutuhkan pergerakan yang konsisten
- gunakan pencahayaan yang cukup
- jaga posisi tangan tetap di dalam frame camera


## AUTHOR 
dibuat oleh Nuril Aisyahroni (masih dibantu guys sama bapak)




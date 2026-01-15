import os
import numpy as np

DATA_PATH = "data/raw"

for gesture in os.listdir(DATA_PATH):
    g_path = os.path.join(DATA_PATH, gesture)
    if not os.path.isdir(g_path):
        continue

    print(f"\nGesture: {gesture}")
    files = os.listdir(g_path)

    for f in files[:3]:  # cek 3 file pertama aja
        data = np.load(os.path.join(g_path, f))
        print(f"  {f} -> shape {data.shape}")
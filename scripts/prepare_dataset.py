import os
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw"
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 126

X = []
y = []
labels = sorted(os.listdir(DATA_PATH))

label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    gesture_path = os.path.join(DATA_PATH, label)

    for file in os.listdir(gesture_path):
        if not file.endswith(".npy"):
            continue

        data = np.load(
            os.path.join(gesture_path, file),
            allow_pickle=True
        )

        # === VALIDASI DATA ===
        if not isinstance(data, np.ndarray):
            print(f"Skip {file} (bukan ndarray)")
            continue

        if data.shape != (SEQUENCE_LENGTH, NUM_KEYPOINTS):
            print(f"Skip {file} shape {data.shape}")
            continue

        # paksa numeric
        data = data.astype(np.float32)

        X.append(data)
        y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Dataset OK")
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("Dataset siap untuk training")
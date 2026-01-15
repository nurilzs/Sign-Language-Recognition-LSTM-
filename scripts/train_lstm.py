import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === LOAD DATA ===
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

num_classes = len(np.unique(y_train))

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("X train:", X_train.shape)
print("y train:", y_train.shape)

# === MODEL ===
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.3),

    LSTM(128),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# === TRAIN ===
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

model.save("models/sign_language_lstm.h5")
print("Model saved")
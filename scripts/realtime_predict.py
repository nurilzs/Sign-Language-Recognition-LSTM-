import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import time
import pygame

# ================= CONFIG =================
MODEL_PATH = "models/sign_language_lstm.h5"
DATA_PATH = "data/raw"
SOUND_DIR = "sounds"

SEQUENCE_LENGTH = 30
THRESHOLD = 0.8
COOLDOWN = 1.2  # detik antar suara

# ================= LOAD LABELS (HARUS SAMA DENGAN TRAINING) =================
GESTURES = sorted(os.listdir(DATA_PATH))

SOUND_MAP = {
    "halo": "Halo.wav",
    "I": "I.wav",
    "L": "L.wav",
    "N": "N.wav",
    "nama": "Nama.wav",
    "R": "R.wav",
    "Salam": "salamkenal.wav",
    "saya": "Saya.wav",
    "U": "U.wav"
}

# ================= INIT AUDIO =================
pygame.mixer.init()

def play_sound(path):
    try:
        pygame.mixer.stop()
        sound = pygame.mixer.Sound(path)
        sound.play()
    except Exception as e:
        print("Audio error:", e)

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = []
current_word = ""
last_word = ""
last_sound_time = 0

# ================= FUNCTIONS =================
def extract_keypoints(results):
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

    while len(keypoints) < 126:
        keypoints.extend([0, 0, 0])

    return np.array(keypoints)

# ================= MAIN LOOP =================
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = sequence[-SEQUENCE_LENGTH:]
            res = model.predict(
                np.expand_dims(sequence, axis=0),
                verbose=0
                )[0]
                
            top2 = np.argsort(res)[-2:]
            best_idx = top2[-1]
            second_idx = top2[-2]
                
            best_conf = res[best_idx]
            second_conf = res[second_idx]
                
            if best_conf > THRESHOLD and (best_conf - second_conf) > 0.2:
                word = GESTURES[best_idx]
                now = time.time()
                    
                if word != last_word and (now - last_sound_time) > COOLDOWN:
                    last_word = word
                    current_word = word
                    last_sound_time = now
                        
                    sound_file = SOUND_MAP.get(word)
                    if sound_file:
                        sound_path = os.path.join(SOUND_DIR, sound_file)
                        if os.path.exists(sound_path):
                            play_sound(sound_path)


        # ================= UI =================
        cv2.rectangle(frame, (0, 0), (450, 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            current_word,
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 255, 0),
            3
        )

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if cv2.getWindowProperty("Sign Language Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

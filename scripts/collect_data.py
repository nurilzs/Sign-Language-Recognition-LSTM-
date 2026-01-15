import cv2
import numpy as np
import mediapipe as mp
import os
import time

# ===================== CONFIG =====================
DATA_PATH = "data/raw"
GESTURE = "Salam"            # ganti tiap gesture
NUM_SAMPLES = 30            # jumlah sequence
SEQUENCE_LENGTH = 30       # frame per sequence
COUNTDOWN_SECONDS = 3
CAMERA_INDEX = 0
# =================================================

# ============ MEDIAPIPE INIT ============
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ============ FOLDER SETUP ============
gesture_path = os.path.join(DATA_PATH, GESTURE)
os.makedirs(gesture_path, exist_ok=True)

# ============ CAMERA ============
cap = cv2.VideoCapture(CAMERA_INDEX)

# ================= FUNCTIONS =================

def extract_keypoints(results):
    """
    Extract 2-hand keypoints (126 values).
    Jika tangan < 2 → padding 0
    """
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

    while len(keypoints) < 126:
        keypoints.extend([0, 0, 0])

    return np.array(keypoints)


def draw_info(frame, text, y=40, color=(255, 255, 255), scale=0.8):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


def countdown(frame, seconds):
    """
    Countdown sebelum recording dimulai
    """
    for i in range(seconds, 0, -1):
        temp = frame.copy()
        cv2.putText(
            temp,
            f"Recording starts in {i}",
            (160, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 0, 255),
            4
        )
        cv2.imshow("Collecting Data", temp)
        cv2.waitKey(1000)

# ================= MAIN LOOP =================

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    for sample_idx in range(NUM_SAMPLES):
        print(f"\n📸 Sample {sample_idx+1}/{NUM_SAMPLES}")

        # ---------- WAIT MODE ----------
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            draw_info(frame, f"Gesture : {GESTURE}")
            draw_info(frame, f"Sample  : {sample_idx+1}/{NUM_SAMPLES}", 80)
            draw_info(frame, "Press SPACE to start", 120, (0, 255, 255))
            draw_info(frame, "Press ESC to quit", 160, (0, 0, 255))

            cv2.imshow("Collecting Data", frame)
            key = cv2.waitKey(1)

            if key == 32:  # SPACE
                countdown(frame, COUNTDOWN_SECONDS)
                break
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # ---------- RECORD MODE ----------
        sequence = []

        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            draw_info(
                frame,
                f"Recording Frame {frame_idx+1}/{SEQUENCE_LENGTH}",
                200,
                (0, 255, 0)
            )

            cv2.imshow("Collecting Data", frame)
            cv2.waitKey(1)

        # ---------- SAVE ----------
        np.save(
            os.path.join(gesture_path, f"{sample_idx}.npy"),
            np.array(sequence)
        )

        print(f"Saved sample {sample_idx}")

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time

# -----------------------
# Constants
# -----------------------
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6  # Mouth open threshold (yawning)
DROWSY_TIME = 2.0  # seconds

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmark indices (MediaPipe)
OUTER_LIPS = [61, 81, 311, 291, 78, 308, 13, 14]  # corners & inner lip points

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

start_drowsy = None
start_yawn = None

def eye_aspect_ratio(eye, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]

    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth]

    # vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[6]) - np.array(pts[7]))
    # horizontal distance
    D = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    return (A + B + C) / (2.0 * D)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0].landmark

        # EAR
        left_ear = eye_aspect_ratio(LEFT_EYE, face, w, h)
        right_ear = eye_aspect_ratio(RIGHT_EYE, face, w, h)
        ear = (left_ear + right_ear) / 2
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # MAR
        mar = mouth_aspect_ratio(OUTER_LIPS, face, w, h)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Drowsiness alert
        if ear < EAR_THRESHOLD:
            if start_drowsy is None:
                start_drowsy = time.time()
            elif time.time() - start_drowsy > DROWSY_TIME:
                cv2.putText(frame, "DROWSINESS ALERT! (Eyes Closed)",
                            (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 3)
        else:
            start_drowsy = None

        # Yawning alert
        if mar > MAR_THRESHOLD:
            if start_yawn is None:
                start_yawn = time.time()
            elif time.time() - start_yawn > 1.0:  # 1 sec continuous yawn
                cv2.putText(frame, "YAWNING ALERT!",
                            (100, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 3)
        else:
            start_yawn = None

    else:
        cv2.putText(frame, "NOT ATTENTIVE (No Face)",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

    cv2.imshow("Drowsiness & Attention Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


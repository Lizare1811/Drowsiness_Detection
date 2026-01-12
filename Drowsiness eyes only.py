#Drowsiness only Eye

import cv2, mediapipe as mp, numpy as np

EAR_THRESH, MAR_THRESH = 0.25, 0.6

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
OUTER_LIPS = [61,81,311,291,78,308,13,14]

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

def ear(eye, lms, w, h):
    pts = [(int(lms[i].x*w), int(lms[i].y*h)) for i in eye]
    return (np.linalg.norm(np.array(pts[1])-np.array(pts[5])) +
            np.linalg.norm(np.array(pts[2])-np.array(pts[4]))) / (2*np.linalg.norm(np.array(pts[0])-np.array(pts[3])))

def mar(mouth, lms, w, h):
    pts = [(int(lms[i].x*w), int(lms[i].y*h)) for i in mouth]
    return (np.linalg.norm(np.array(pts[1])-np.array(pts[5])) +
            np.linalg.norm(np.array(pts[2])-np.array(pts[4])) +
            np.linalg.norm(np.array(pts[6])-np.array(pts[7]))) / (2*np.linalg.norm(np.array(pts[0])-np.array(pts[3])))

current_state = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    state = "No Face"  # default

    if res.multi_face_landmarks:
        f = res.multi_face_landmarks[0].landmark
        eye_ratio = (ear(LEFT_EYE,f,w,h)+ear(RIGHT_EYE,f,w,h))/2
        mouth_ratio = mar(OUTER_LIPS,f,w,h)

        if eye_ratio < EAR_THRESH:
            state = "Drowsy"
        elif mouth_ratio > MAR_THRESH:
            state = "Yawning"
        else:
            state = "Attentive"

    # Print and display only if state changes
    if state != current_state:
        current_state = state
        print(f"Person is {state}")

    # Show on screen
    if state == "Drowsy":
        cv2.putText(frame,"DROWSINESS ALERT!",(50,50),1,0.7,(0,0,255),2)
    elif state == "Yawning":
        cv2.putText(frame,"YAWNING ALERT!",(50,50),1,0.7,(0,0,255),2)
    else:
        cv2.putText(frame,"Alert",(50,50),1,0.7,(0,255,0),2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release()
cv2.destroyAllWindows()

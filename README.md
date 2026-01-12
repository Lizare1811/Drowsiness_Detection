Drowsiness Detection System

This project detects driver drowsiness in real-time using a webcam. It includes two detection methods:
1)Eye-only detection: Uses Eye Aspect Ratio (EAR) to detect if the eyes are closed.

2)EAR + MAR detection: Uses Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR) to detect both eye closure and yawning.

Features
Real-time drowsiness detection using OpenCV and MediaPipe.
Alerts the user when drowsiness is detected.
Supports two modes:
-Eye-only: Simple eye closure detection.
-EAR + MAR: Eye + Mouth detection for better accuracy.

Project Structure
-Drowsiness_Detection
-drowsiness_eye_only.py       
-drowsiness_ear_mar.py        
-README.md                   

Note: No dataset/images are required as detection is done in real-time via webcam.

How to Run

1)Install dependencies:
pip install opencv-python mediapipe numpy

2)Run Eye-only detection:
python drowsiness_eye_only.py

3)Run EAR + MAR detection:
python drowsiness_ear_mar.py


What happens:
The webcam opens and tracks the face.
EAR threshold is used to detect closed eyes.
MAR threshold is used to detect yawning (in EAR+MAR mode).
Alerts are displayed on screen if drowsiness is detected.

Thresholds Used
EAR threshold: 0.25 (for eye closure)
MAR threshold: 0.6 (for yawning)
-These thresholds can be adjusted depending on lighting and user behavior.

Future Enhancements
Add sound or SMS alerts when drowsiness is detected.
Combine with driver fatigue database to improve accuracy.
Add support for multiple faces in the frame.

Dependencies
OpenCV (opencv-python)
MediaPipe (mediapipe)
NumPy (numpy)

Drowsiness Detection System

This project detects driver drowsiness in real-time using a webcam. It includes two detection methods:

1)Eye-only detection: Uses Eye Aspect Ratio (EAR) to detect if the eyes are closed.

2)EAR + MAR detection: Uses Eye Aspect Ratio (EAR) + Mouth Aspect Ratio (MAR) to detect both eye closure and yawning.

Features:

1)Real-time drowsiness detection using OpenCV and MediaPipe.

2)Alerts the user when drowsiness is detected.

3)Supports two modes:-Eye-only: Simple eye closure detection. -EAR + MAR: Eye + Mouth detection for better accuracy.

Project Structure

1)Drowsiness_Detection

2)drowsiness_eye_only.py   

3)drowsiness_ear_mar.py    

4)README.md                   

Note: No dataset/images are required as detection is done in real-time via webcam.

How to Run

1)Install dependencies:
pip install opencv-python mediapipe numpy

2)Run Eye-only detection:
python drowsiness_eye_only.py

3)Run EAR + MAR detection:
python drowsiness_ear_mar.py


What happens:

1)The webcam opens and tracks the face.
2)EAR threshold is used to detect closed eyes.
3)MAR threshold is used to detect yawning (in EAR+MAR mode).
4)Alerts are displayed on screen if drowsiness is detected.

Thresholds Used
1)EAR threshold: 0.25 (for eye closure)

2)MAR threshold: 0.6 (for yawning)

These thresholds can be adjusted depending on lighting and user behavior.

Future Enhancements

1)Add sound or SMS alerts when drowsiness is detected.

2)Combine with driver fatigue database to improve accuracy.

3)Add support for multiple faces in the frame.

Dependencies

1)OpenCV (opencv-python)
2)MediaPipe (mediapipe)
3)NumPy (numpy)

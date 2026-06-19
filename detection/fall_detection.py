import cv2
import mediapipe as mp
import math
import csv
import os
from datetime import datetime

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

csv_file = "data/fall_logs.csv"

if not os.path.exists(csv_file):

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "event"
            ]
        )

fall_detected = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (640, 480)
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    status = "Normal"

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        shoulder_y = (
            left_shoulder.y +
            right_shoulder.y
        ) / 2

        hip_y = (
            left_hip.y +
            right_hip.y
        ) / 2

        vertical_distance = abs(
            shoulder_y - hip_y
        )

        if vertical_distance < 0.12:

            status = "FALL DETECTED"

            if not fall_detected:

                fall_detected = True

                with open(
                    csv_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow(
                        [
                            datetime.now(),
                            "Fall Detected"
                        ]
                    )

        else:

            fall_detected = False

    color = (
        (0, 0, 255)
        if status == "FALL DETECTED"
        else (0, 255, 0)
    )

    cv2.putText(
        frame,
        status,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow(
        "Fall Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
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

csv_file = "data/activity_logs.csv"

if not os.path.exists(csv_file):

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "timestamp",
                "activity",
                "fall_detected"
            ]
        )

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1]-b[1], c[0]-b[0])
        -
        math.atan2(a[1]-b[1], a[0]-b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle

def detect_activity(landmarks):

    shoulder = landmarks[11]
    hip = landmarks[23]
    knee = landmarks[25]

    shoulder_point = (
        shoulder.x,
        shoulder.y
    )

    hip_point = (
        hip.x,
        hip.y
    )

    knee_point = (
        knee.x,
        knee.y
    )

    angle = calculate_angle(
        shoulder_point,
        hip_point,
        knee_point
    )

    if angle > 150:
        return "Standing"

    elif angle < 100:
        return "Sitting"

    elif abs(shoulder.y - hip.y) < 0.15:
        return "Lying"

    else:
        return "Walking"

last_activity = ""

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

    activity = "Unknown"

    fall_detected = "No"

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        activity = detect_activity(
            landmarks
        )

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

            if activity == "Lying":
                fall_detected = "Yes"
            else:
                fall_detected = "No"

        if activity != last_activity:

            with open(
                csv_file,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow(
                    [
                        datetime.now(),
                        activity,
                        fall_detected
                    ]
                )

            print(
                f"{activity} | Fall:{fall_detected}"
            )

            last_activity = activity

    cv2.putText(
        frame,
        f"Activity: {activity}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Fall: {fall_detected}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.imshow(
        "Activity Logger",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
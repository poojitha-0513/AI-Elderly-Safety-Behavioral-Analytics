import cv2
import mediapipe as mp
import math
import joblib
import pandas as pd
import numpy as np

model = joblib.load(
    "models/risk_model.pkl"
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

standing_count = 0
walking_count = 0
sitting_count = 0
lying_count = 0

fall_count = 0

inactivity_alerts = 0
lying_alerts = 0

fall_triggered = False

def calculate_angle(a,b,c):

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

    elif abs(shoulder.y-hip.y) < 0.15:
        return "Lying"

    else:
        return "Walking"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (900,600)
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb)

    activity = "Unknown"
    risk_level = "Low"
    alert_message = ""

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

        if activity == "Standing":
            standing_count += 1

        elif activity == "Walking":
            walking_count += 1

        elif activity == "Sitting":
            sitting_count += 1

        elif activity == "Lying":
            lying_count += 1

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

            alert_message = "FALL DETECTED"

            if not fall_triggered:

                fall_count += 1
                fall_triggered = True

        else:

            fall_triggered = False

        if activity == "Lying":
            lying_alerts = lying_count // 100

        if activity == "Sitting":
            inactivity_alerts = sitting_count // 150

        movement_score = (
            walking_count * 5
            +
            standing_count * 2
        )

        feature_df = pd.DataFrame(
            [{
                "standing_time":
                    standing_count,

                "walking_time":
                    walking_count,

                "sitting_time":
                    sitting_count,

                "lying_time":
                    lying_count,

                "fall_count":
                    fall_count,

                "inactivity_alerts":
                    inactivity_alerts,

                "lying_alerts":
                    lying_alerts,

                "movement_score":
                    movement_score
            }]
        )

        prediction = model.predict(
            feature_df
        )

        risk_level = prediction[0]

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
        f"Risk: {risk_level}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Falls: {fall_count}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.putText(
        frame,
        alert_message,
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.imshow(
        "AI Elderly Safety Monitoring System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
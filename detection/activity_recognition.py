import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected")
    exit()

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle

def detect_activity(landmarks):

    shoulder = landmarks[11]
    hip = landmarks[23]
    knee = landmarks[25]
    ankle = landmarks[27]
    
    shoulder_point = (shoulder.x, shoulder.y)
    hip_point = (hip.x, hip.y)
    knee_point = (knee.x, knee.y)
    ankle_point = (ankle.x, ankle.y)

    body_angle = calculate_angle(
        shoulder_point,
        hip_point,
        knee_point
    )

    if body_angle > 150:
        return "Standing"

    elif body_angle < 100:
        return "Sitting"

    elif abs(shoulder.y - hip.y) < 0.15:

        return "Lying Down"
    
    else:
        return "Walking"

previous_activity = ""

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = pose.process(rgb)

    activity = "No Person"

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        activity = detect_activity(landmarks)
        if activity != previous_activity:

            print("Activity:", activity)

            previous_activity = activity

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.putText(
        frame,
        f"Activity: {activity}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Activity Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
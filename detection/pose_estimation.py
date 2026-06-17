import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640,480))

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(rgb_frame)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        nose = landmarks[0]

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        left_hip = landmarks[23]
        right_hip = landmarks[24]

        left_knee = landmarks[25]
        right_knee = landmarks[26]

        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        keypoints = {

            "nose":
            (
                nose.x,
                nose.y
            ),

            "left_shoulder":
            (
                 left_shoulder.x,
                 left_shoulder.y
            ),

            "right_shoulder":
            (
                right_shoulder.x,
                right_shoulder.y
            ),

            "left_hip":
            (
                left_hip.x,
            left_hip.y
            ),

            "right_hip":
            (
                right_hip.x,
                right_hip.y
            )
        }

        print(keypoints)

        cv2.putText(
            frame,
            f"Nose: {nose.x:.2f}, {nose.y:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing_utils = mp.solutions.drawing_utils

def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    result = model.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, result

def draw_landmarks(frame, result):
    mp_drawing_utils.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                    mp_drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                    mp_drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                    )
    mp_drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                    mp_drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                    )
    mp_drawing_utils.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                    mp_drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                    )
    mp_drawing_utils.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                    mp_drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                    )

def extract_landmark(result):
    pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in result.pose_landmarks.landmark]).flatten()\
        if result.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.face_landmarks.landmark]).flatten() \
        if result.face_landmarks else np.zeros(468 * 3)
    left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.left_hand_landmarks.landmark]).flatten() \
        if result.left_hand_landmarks else np.zeros(21 * 3)
    right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in result.right_hand_landmarks.landmark]).flatten() \
        if result.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, left_hand, right_hand])

capture = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        return_value, frame = capture.read()
        frame, result = mediapipe_detection(frame, holistic)
        draw_landmarks(frame, result)
        cv2.imshow("Mime by Ridhi and Kartikay", frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
    capture.release()
    cv2.destroyAllWindows()
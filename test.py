import cv2
import mediapipe as mp
import numpy as np
import time
from os import path

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

window_title = "Custom Gesture Detector"
hands_count = 2
is_flip = True

width = 1280
height = 720

MODEL_HAND_GESTURE = "model/gesture/gesture_recognizer.task"

LABEL_NO_CAMERA = "Please check your video capturing device."
LABEL_FPS = "FPS: %.1f"

MARGIN = 10 
FONT_SIZE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)
XY_FPS = (50, 50)


base_options = python.BaseOptions(model_asset_path=MODEL_HAND_GESTURE, delegate="GPU")
hand_options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=hands_count,
    running_mode=vision.RunningMode.VIDEO,
)
recognizer = vision.GestureRecognizer.create_from_options(hand_options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    print(LABEL_NO_CAMERA)
    cap.release()
    exit()

frame_index = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from camera!")
        break

    cost_start = time.time()
    frame_index += 1
    frame_timestamp_ms = int(round(1000 * frame_index / fps))


    if is_flip:
        image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    results_gesture = recognizer.recognize_for_video(mp_image, frame_timestamp_ms)


    annotated_image = np.copy(rgb_image)

    # Results contain:
    #   results_gesture.gestures         -> list of recognized gestures
    #   results_gesture.hand_landmarks   -> list of landmarks
    #   results_gesture.handedness       -> "Left"/"Right"
    hand_landmarks_list = results_gesture.hand_landmarks
    handedness_list = results_gesture.handedness
    gestures_list = results_gesture.gestures

    # Draw each hand’s landmarks + recognized gesture text
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        gesture = gestures_list[idx] if gestures_list else None

        
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])

        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )


        height_i, width_i, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * width_i)
        text_y = int(min(y_coords) * height_i) - MARGIN

        current_hand = handedness[0].category_name
        if is_flip:
            if current_hand == "Left":
                current_hand = "Right"
            elif current_hand == "Right":
                current_hand = "Left"

        gesture_label = gesture[0].category_name if gesture else "Unknown"
        cv2.putText(
            annotated_image,
            f"{current_hand} {gesture_label}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            (0, 255, 0),
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    # Show FPS
    time_consuming = time.time() - cost_start
    cur_fps = 1.0 / time_consuming if time_consuming > 0 else 0
    cv2.putText(
        annotated_image,
        LABEL_FPS % cur_fps,
        XY_FPS,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SIZE,
        TEXT_COLOR,
        FONT_THICKNESS
    )

  # Conver back to bgrrrr
    output_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_title, output_frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('f'): 
        is_flip = not is_flip

cap.release()
cv2.destroyAllWindows()

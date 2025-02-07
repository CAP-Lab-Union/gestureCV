import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class GestureDetector:
    def __init__(self, model_path: str, max_hands: int = 2):
        self.max_hands = max_hands
        self.is_flip = True 
    
        self.MARGIN = 10
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 2
        self.TEXT_COLOR = (0, 255, 0)  
        
        # Initialize MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_path, delegate="GPU")
        self.options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=self.max_hands,
            running_mode=vision.RunningMode.VIDEO  
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def process_frame(self, frame: np.ndarray, frame_timestamp_ms: int):
        """
        Process a single frame and return gesture recognition results with annotations
        """
        if self.is_flip:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform gesture recognition
        results_gesture = self.recognizer.recognize_for_video(mp_image, frame_timestamp_ms)
        
        # Create annotated image
        annotated_image = np.copy(rgb_frame)
        
        # Process results
        processed_results = self._process_and_draw_results(
            results_gesture, 
            annotated_image
        )
        
        # Convert back to BGR for web display
        output_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        return processed_results, output_frame

    def _process_and_draw_results(self, results_gesture, annotated_image):
        """
        Process results and draw landmarks and text on the image
        """
        processed_results = []
        height_i, width_i, _ = annotated_image.shape
        
        if results_gesture.hand_landmarks:
            for idx in range(len(results_gesture.hand_landmarks)):
                hand_landmarks = results_gesture.hand_landmarks[idx]
                handedness = results_gesture.handedness[idx]
                gesture = results_gesture.gestures[idx] if results_gesture.gestures else None

                # Convert landmarks to proto format for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

                # Calculate text position
                x_coords = [lm.x for lm in hand_landmarks]
                y_coords = [lm.y for lm in hand_landmarks]
                text_x = int(min(x_coords) * width_i)
                text_y = int(min(y_coords) * height_i) - self.MARGIN

                # Handle hand laterality
                current_hand = handedness[0].category_name
                if self.is_flip:
                    current_hand = "Right" if current_hand == "Left" else "Left"

                # Get gesture label
                gesture_label = gesture[0].category_name if gesture else "Unknown"

                # Draw text
                cv2.putText(
                    annotated_image,
                    f"{current_hand} {gesture_label}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.FONT_SIZE,
                    self.TEXT_COLOR,
                    self.FONT_THICKNESS,
                    cv2.LINE_AA
                )

                # Store results
                result = {
                    "hand_index": idx,
                    "handedness": current_hand,
                    "gesture": gesture_label,
                    "landmarks": [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                    "text_position": (text_x, text_y)
                }
                if gesture:
                    result["gesture_score"] = float(gesture[0].score)
                
                processed_results.append(result)

        return processed_results

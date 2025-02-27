import os
import base64
import cv2
import numpy as np
import shutil
from mediapipe_model_maker import gesture_recognizer

def decode_image(base64_str, frame_width=224, frame_height=224):
    """Decode base64 image string and resize to specified dimensions."""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed.")
        return cv2.resize(img, (frame_width, frame_height))
    except Exception as e:
        raise ValueError(f"Error decoding image: {e}")


def train_model_real_time(training_images, gesture_name, learning_rate=0.001, epochs=5, batch_size=1):
    persistent_dataset_dir = os.path.join(os.getcwd(), "gestureDataset")  
    export_dir = os.path.join(os.getcwd(), "gestureModels")
    os.makedirs(persistent_dataset_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # Gesture-specific folder
    gesture_dir = os.path.join(persistent_dataset_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)


    none_dir = os.path.join(persistent_dataset_dir, "None")
    if not os.path.exists(none_dir) or len(os.listdir(none_dir)) < 10:
        os.makedirs(none_dir, exist_ok=True)
        for idx in range(10):
            blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(none_dir, f"none_{idx}.jpg"), blank_img)

    valid_images = 0
    for idx, img_b64 in enumerate(training_images):
        try:
            img = decode_image(img_b64)
            cv2.imwrite(os.path.join(gesture_dir, f"gesture_{idx}.jpg"), img)
            valid_images += 1
        except Exception as e:
            print(f"Skipping invalid image: {e}")
            continue

    if valid_images < 10:
        raise ValueError(f"At least 10 valid images required, got {valid_images}")

    data = gesture_recognizer.Dataset.from_folder(
        dirname=persistent_dataset_dir,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    train_data, test_data = data.split(0.9)

    hparams = gesture_recognizer.HParams(
        export_dir=export_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=test_data,
        options=options
    )

    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss: {loss}, Test accuracy: {acc}")

    model_path = os.path.join(export_dir, "gesture_recognizer_incremental.task")
    model.export_model(model_path)
    print(f"Model exported to: {model_path}")
    return model_path  # TODO: For frontend/async feedback
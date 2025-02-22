import os
import uuid
import datetime
from mediapipe_model_maker import gesture_recognizer
#
# mounted as a Docker volume
GESTURE_DATASET_DIR = "gesture_imgs"


def register_gesture(gesture_name: str) -> str:
    """
    Ensure that a directory exists for the given gesture.
    """
    gesture_dir = os.path.join(GESTURE_DATASET_DIR, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
    return gesture_dir

def save_gesture_capture(gesture_name: str, image_bytes: bytes) -> str:
    """
    Save the uploaded image bytes for the given gesture.
    """
    gesture_dir = register_gesture(gesture_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{gesture_name}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    file_path = os.path.join(gesture_dir, filename)
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return file_path

def get_capture_instructions():
    """
    Returns a list of instructions to guide the user in capturing images
    from different angles.
    """
    instructions = [
        "Capture the front view of the gesture.",
        "Capture the left angle of the gesture.",
        "Capture the right angle of the gesture.",
        "Capture the gesture from a slightly elevated angle.",
        "Capture the gesture from a slightly lower angle."
    ]
    return instructions

def train_gesture_recognizer(
    learning_rate: float = 0.001,
    epochs: int = 30,
    batch_size: int = 1,
    validation_batch_size: int = 1,
    export_dir: str = "./exported_models",
    export_model_name: str = "gesture_recognizer"
):
    """
    Load all the gesture image data from the dataset directory,
    train the gesture recognizer, evaluate it, and export the model.
    """
    labels = []
    for item in os.listdir(GESTURE_DATASET_DIR):
        if os.path.isdir(os.path.join(GESTURE_DATASET_DIR, item)):
            labels.append(item)
    print("Labels found:", labels)

    data = gesture_recognizer.Dataset.from_folder(
        dirname=GESTURE_DATASET_DIR,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    hparams = gesture_recognizer.HParams(
        export_dir=export_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    loss, acc = model.evaluate(test_data, batch_size=validation_batch_size)
    print(f"Test loss: {loss}, Test accuracy: {acc}")

    model_path = os.path.join(export_dir, export_model_name + ".task")
    model.export_model(model_name=export_model_name + ".task")
    print(f"Model exported to: {model_path}")

    return {
        "labels": labels,
        "loss": loss,
        "accuracy": acc,
        "model_path": model_path
    }
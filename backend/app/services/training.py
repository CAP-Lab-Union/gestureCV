# import os
# import base64
# import cv2
# import numpy as np
# import tensorflow as tf
# from mediapipe_model_maker import gesture_recognizer




# def decode_image(base64_str, frame_width=224, frame_height=224):
#     """"Decode the base64 image string and resize it to the specified dimensions."""
#     try:
#         if ',' in base64_str:
#             base64_str = base64_str.split(',')[1]
#         img_data = base64.b64decode(base64_str)
#         nparr = np.frombuffer(img_data, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError("Image decoding failed.")
#         img_resized = cv2.resize(img, (frame_width, frame_height))
#         return img_resized
#     except Exception as e:
#         print(f"Error decoding image: {e}")
#         return None

# def train_model(training_images: list[str], gesture_label: int):
#     """
#     Accepts a list of base64-encoded images and a gesture label,
#     decodes and preprocesses the images, trains a simple CNN, and saves the model.
#     """
#     captured_images = []
#     for img_b64 in training_images:
#         img = decode_image(img_b64)
#         if img is not None:
#             captured_images.append(img)
    
#     if not captured_images:
#         raise ValueError("No valid images provided for training.")
    
#     captured_images = np.array(captured_images, dtype="float32") / 255.0
#     print(f"Training on {len(captured_images)} images.")

#     labels = np.full((len(captured_images),), gesture_label)

# #cnn here 
#     num_classes = 1
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dense(num_classes, activation="sigmoid")
#     ])
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     model.summary()
#     model.fit(captured_images, labels, epochs=5, batch_size=8)


#     # Below Is the savoinmgm of the model
#     save_dir = os.path.join(os.getcwd(), "gestureModels")
#     os.makedirs(save_dir, exist_ok=True)  
#     h5_path = os.path.join(save_dir, "gesture_real_time.h5")
#     tflite_path = os.path.join(save_dir, "gesture_real_time.tflite")
    
#     # Save the model in H5 format
#     model.save(h5_path)
#     print(f"Updated model saved as {h5_path}")
    
#     # Convert to TFLite and save
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     tflite_model = converter.convert()
#     with open(tflite_path, "wb") as f:
#         f.write(tflite_model)
#     print(f"TFLite model saved as {tflite_path}")

#     # Cleanup
#     del captured_images, labels
#     print("Temporary data cleared.")

# # =============================================
# # TODO: MediaPipe Model Maker Implementation
# # =============================================

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

def train_model(training_images: list[str], gesture_name: str,
                learning_rate: float = 0.001, epochs: int = 30, 
                batch_size: int = 1, validation_batch_size: int = 1):
    """
    Accepts base64-encoded images and trains a gesture recognition model.
    """
    if not gesture_name:
        raise ValueError("Gesture name must be provided for training.")

    # Create temporary dataset structure
    temp_dataset_dir = os.path.join(os.getcwd(), "gestureDataset_temp")
    shutil.rmtree(temp_dataset_dir, ignore_errors=True)  # Clean previous runs
    
    # Create class directories
    gesture_dir = os.path.join(temp_dataset_dir, gesture_name)
    none_dir = os.path.join(temp_dataset_dir, "None")
    os.makedirs(gesture_dir, exist_ok=True)
    os.makedirs(none_dir, exist_ok=True)

    # Save valid gesture images
    valid_images = 0
    for idx, img_b64 in enumerate(training_images):
        try:
            img = decode_image(img_b64)
            cv2.imwrite(os.path.join(gesture_dir, f"gesture_{idx}.jpg"), img)
            valid_images += 1
        except Exception as e:
            print(f"Skipping invalid image: {e}")
            continue

    if valid_images < 10:  # Minimum samples check
        raise ValueError(f"At least 10 valid images required, got {valid_images}")

    # Add real negative samples to "None" class (critical fix)
    # Replace this with actual background/negative images if available
    # Here we create simple blank images as placeholder negatives
    for i in range(max(1, valid_images // 5)):  # 20% of gesture images
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(none_dir, f"none_{i}.jpg"), blank)

    # Load and verify dataset
    try:
        data = gesture_recognizer.Dataset.from_folder(
            dirname=temp_dataset_dir,
            hparams=gesture_recognizer.HandDataPreprocessingParams()
        )
    except ValueError as e:
        raise RuntimeError(f"Dataset error: {e}. Check folder structure and images.")

    # Improved dataset splitting
    train_data, test_data = data.split(0.9)
    validation_data, test_data = test_data.split(0.5)

    # Configure training
    export_dir = os.path.join(os.getcwd(), "gestureModels")
    os.makedirs(export_dir, exist_ok=True)
    
    hparams = gesture_recognizer.HParams(
        export_dir=export_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
    )
    
    options = gesture_recognizer.GestureRecognizerOptions(
        hparams=hparams,
        model_options=gesture_recognizer.ModelOptions(dropout_rate=0.3)
    )

    # Train model
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Evaluate and export
    evaluation = model.evaluate(test_data, batch_size=validation_batch_size)
    print(f"Model evaluation results: {evaluation}")
    
    export_name = f"gesture_model_{gesture_name}.task"
    model.export_model(os.path.join(export_dir, export_name))
    
    # Cleanup
    shutil.rmtree(temp_dataset_dir)
    return os.path.join(export_dir, export_name)
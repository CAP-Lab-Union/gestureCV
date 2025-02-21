import os
import base64
import cv2
import numpy as np
import tensorflow as tf

def decode_image(base64_str, frame_width=224, frame_height=224):
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image decoding failed.")
        img_resized = cv2.resize(img, (frame_width, frame_height))
        return img_resized
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def train_model(training_images: list[str], gesture_label: int):
    """
    Accepts a list of base64-encoded images and a gesture label,
    decodes and preprocesses the images, trains a simple CNN, and saves the model.
    """
    captured_images = []
    for img_b64 in training_images:
        img = decode_image(img_b64)
        if img is not None:
            captured_images.append(img)
    
    if not captured_images:
        raise ValueError("No valid images provided for training.")
    
    captured_images = np.array(captured_images, dtype="float32") / 255.0
    print(f"Training on {len(captured_images)} images.")

    labels = np.full((len(captured_images),), gesture_label)

#cnn here 
    num_classes = 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(captured_images, labels, epochs=5, batch_size=8)



# TODO: REDO THIS PART CUZ MEDIAPIPE MODEL MAKER WORKS NOW.

    # Below Is the savoinmgm of the model
    save_dir = os.path.join(os.getcwd(), "gestureModels")
    os.makedirs(save_dir, exist_ok=True)  
    h5_path = os.path.join(save_dir, "gesture_real_time.h5")
    tflite_path = os.path.join(save_dir, "gesture_real_time.tflite")
    
    # Save the model in H5 format
    model.save(h5_path)
    print(f"Updated model saved as {h5_path}")
    
    # Convert to TFLite and save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved as {tflite_path}")

    # Cleanup
    del captured_images, labels
    print("Temporary data cleared.")
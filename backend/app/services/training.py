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
import tensorflow as tf
import shutil
from mediapipe_model_maker import gesture_recognizer

def decode_image(base64_str, frame_width=224, frame_height=224):

    """"Decode the base 64 image string and resize it to the specified dimensions."""
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
                learning_rate: float = 0.001, epochs: int = 5, batch_size: int = 8,
                validation_batch_size: int = 8):
    """""Accepts a list of base64-encoded images and a gesture name,decodes and preprocesses the images, trains a simple CNN, and saves the model."""
    temp_dataset_dir = os.path.join(os.getcwd(), "gestureDataset_temp")
    os.makedirs(temp_dataset_dir, exist_ok=True)
    
    none_dir = os.path.join(temp_dataset_dir, "None")
    os.makedirs(none_dir, exist_ok=True)

    # This the gesture name below, custom_folder is the folder where the images
    custom_folder = os.path.join(temp_dataset_dir, f"z_{gesture_name}")
    os.makedirs(custom_folder, exist_ok=True)

    # TODO: The labeling issue is below, (line 126 - 134). 
    valid_images = 0
    for idx, img_b64 in enumerate(training_images):
        try:
            img = decode_image(img_b64)
            img_filename = os.path.join(custom_folder, f"img_{idx}.jpg")
            cv2.imwrite(img_filename, img)
            valid_images += 1
        except Exception:
            continue

    # labels = [] 
    # valid_images = 0
    # for idx, img_b64 in enumerate(training_images):
    #     img = decode_image(img_b64)
    #     img_filename = os.path.join(custom_folder, f"img_{idx}.jpg")
    #     cv2.imwrite(img_filename, img)
    #     valid_images += 1 

    if valid_images == 0:
        raise ValueError("No valid images provided for training.")


    if not os.listdir(none_dir):
        placeholder = os.path.join(custom_folder, os.listdir(custom_folder)[0])
        cv2.imwrite(os.path.join(none_dir, "img_none.jpg"), cv2.imread(placeholder))

    data = gesture_recognizer.Dataset.from_folder(
        dirname=temp_dataset_dir,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    export_dir = os.path.join(os.getcwd(), "gestureModels")
    os.makedirs(export_dir, exist_ok=True)

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
    
    _ = model.evaluate(test_data, batch_size=validation_batch_size)
    

    export_name = f"gesture_model_{gesture_name}"
    model.export_model(model_name=export_name + ".task")
    
    shutil.rmtree(temp_dataset_dir)

import os
from dotenv import load_dotenv

load_dotenv()


DEBUG = os.getenv("DEBUG", "False") == "True"
MODEL_PATH = os.getenv("MODEL_PATH", "model/gesture/gesture_recognizer.task")
MAX_HANDS = int(os.getenv("MAX_HANDS", "20"))

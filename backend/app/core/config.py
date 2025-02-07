from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Gesture Detection API"
    MODEL_PATH: str = "model/gesture/gesture_recognizer.task"
    CORS_ORIGINS: list = ["http://localhost:3000"] 
    
    class Config:
        case_sensitive = True

settings = Settings()
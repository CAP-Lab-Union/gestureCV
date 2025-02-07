from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str
    CAMERA_WIDTH: int
    CAMERA_HEIGHT: int
    
    class Config:
        env_file = ".env"

settings = Settings()
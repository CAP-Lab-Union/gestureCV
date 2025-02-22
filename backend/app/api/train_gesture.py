from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Body
from pydantic import BaseModel
from ..services.train_services import (
    save_gesture_capture,
    get_capture_instructions,
    train_gesture_recognizer
)

# Pydantic model for training parameters
class TrainParams(BaseModel):
    learning_rate: float = 0.001
    epochs: int = 30
    batch_size: int = 1
    validation_batch_size: int = 1
    export_dir: str = "./exported_models"
    export_model_name: str = "gesture_recognizer"

router = APIRouter()

@router.post("/capture")
async def capture_gesture(gesture_name: str = Form(...), file: UploadFile = File(...)):
    """
    Endpoint to capture a gesture image.
    The image is saved to the corresponding gesture folder.
    """
    image_bytes = await file.read()
    file_path = save_gesture_capture(gesture_name, image_bytes)
    return {"message": "Image captured", "file_path": file_path}

@router.get("/instructions")
def instructions():
    """
    Endpoint to provide capture instructions for different angles.
    """
    instructions = get_capture_instructions()
    return {"instructions": instructions}

@router.post("/train")
def train_model(
    params: TrainParams = Body(default_factory=TrainParams),
    background_tasks: BackgroundTasks = Body(...)
):
    """
    Endpoint to trigger training.
    Training is started as a background task so the API returns immediately.
    """
    background_tasks.add_task(
        train_gesture_recognizer,
        learning_rate=params.learning_rate,
        epochs=params.epochs,
        batch_size=params.batch_size,
        validation_batch_size=params.validation_batch_size,
        export_dir=params.export_dir,
        export_model_name=params.export_model_name
    )
    return {"message": "Training started in background."}

# from fastapi import APIRouter, BackgroundTasks
# from pydantic import BaseModel
# from ..services.training import train_model

# router = APIRouter()

# class TrainingPayload(BaseModel):
#      # list of base64 encoded images
#      # the label for these images
#     training_images: list[str]  
#     gesture_label: int         

# @router.post("/train")
# async def train_endpoint(payload: TrainingPayload, background_tasks: BackgroundTasks):
#     background_tasks.add_task(train_model, payload.training_images, payload.gesture_label)
#     return {"status": "Training started"}

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from ..services.training import train_model  # updated function import

router = APIRouter()

class TrainingPayload(BaseModel):
    training_images: list[str]  # list of base64 encoded images
    gesture_name: str           # the gesture name provided by the frontend

@router.post("/train")
async def train_endpoint(payload: TrainingPayload, background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model, payload.training_images, payload.gesture_name)
    return {"status": "Training started"}

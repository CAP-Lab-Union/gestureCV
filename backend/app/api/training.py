
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from ..services.training import train_model_real_time  

router = APIRouter(prefix="/gesture", tags=["gesture_training"])

class TrainModelRequest(BaseModel):
    training_images: List[str]  
    gesture_name: str
    learning_rate: float = 0.001
    epochs: int = 5
    batch_size: int = 1

# Synchronous endpoint (for quick testing)
@router.post("/train/sync")
async def train_model_sync(req: TrainModelRequest):
    try:
        model_path = train_model_real_time(
            training_images=req.training_images,
            gesture_name=req.gesture_name,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            batch_size=req.batch_size
        )
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_path": model_path
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Asynchronous endpoint (for non-blocking training)
@router.post("/train/async")
async def train_model_async(req: TrainModelRequest, background_tasks: BackgroundTasks):
    try:
        # Validate input upfront
        if len(req.training_images) < 5:
            raise ValueError(f"At least 5 images required, got {len(req.training_images)}")
        
        # Queue training in the background
        background_tasks.add_task(
            train_model_real_time,
            training_images=req.training_images,
            gesture_name=req.gesture_name,
            learning_rate=req.learning_rate,
            epochs=req.epochs,
            batch_size=req.batch_size
        )
        return {
            "status": "accepted",
            "message": "Training started in the background"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
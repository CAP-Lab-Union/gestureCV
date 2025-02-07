from pydantic import BaseModel
from typing import List, Optional

class GestureData(BaseModel):
    hand: str
    gesture: str
    confidence: float
    landmarks: List[dict]

class GestureResponse(BaseModel):
    gestures: List[GestureData]
    frame_data: str
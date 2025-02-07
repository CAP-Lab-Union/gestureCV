from fastapi import WebSocket
from ..services.gesture_detector import GestureDetector  
import base64
import cv2
import numpy as np
from ..core.settings import MODEL_PATH, MAX_HANDS

class GestureWebSocket:
    def __init__(self):
        self.detector = GestureDetector(MODEL_PATH, MAX_HANDS)
        self.active_connections: list[WebSocket] = []
        self.frame_index = 0
        self.fps = 30  # Default FPS

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def process_frame(self, websocket: WebSocket, base64_frame: str):
        """
        Process a single frame and send results back through the WebSocket.
        """
        try:
            encoded_data = (
                base64_frame.split(',')[1]
                if ',' in base64_frame
                else base64_frame
            )
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

   
            self.frame_index += 1
            frame_timestamp_ms = int(round(1000 * self.frame_index / self.fps))

            results, annotated_frame = self.detector.process_frame(
                frame, frame_timestamp_ms
            )

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_json({
                "status": "success",
                "gestures": results,
                "processed_frame": f"data:image/jpeg;base64,{processed_frame_base64}"
            })
        except Exception as e:
            await websocket.send_json({
                "status": "error",
                "message": str(e)
            })

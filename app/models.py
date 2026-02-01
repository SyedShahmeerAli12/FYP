from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class ViolationCreate(BaseModel):
    camera_id: int
    detection_class: str
    confidence: float
    video_path: str
    priority: Optional[str] = None
    timestamp: datetime
    frame_count: Optional[int] = 0
    bbox_data: Optional[Dict] = {}

class ViolationResponse(BaseModel):
    id: int
    camera_id: int
    detection_class: str
    confidence: float
    video_path: str
    priority: Optional[str] = None
    timestamp: str
    frame_count: int
    bbox_data: Dict
    created_at: str


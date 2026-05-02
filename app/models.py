from pydantic import BaseModel, EmailStr
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


class LoginRequest(BaseModel):
    email: str
    password: str
    remember_me: bool = False


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class AddAdminRequest(BaseModel):
    email: EmailStr
    full_name: str


class SetPasswordRequest(BaseModel):
    token: str
    password: str


class GoogleAuthRequest(BaseModel):
    credential: str

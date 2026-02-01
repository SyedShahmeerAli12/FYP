from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import uvicorn
import json
import asyncio
from datetime import datetime, timedelta
import os

from app.detection_service import DetectionService
from app.database import Database
from app.models import ViolationCreate, ViolationResponse
from app.auth import verify_password, get_password_hash, create_access_token, verify_token, generate_reset_token, ACCESS_TOKEN_EXPIRE_HOURS, REMEMBER_ME_EXPIRE_DAYS

app = FastAPI(title="INFOCAM Detection API")

# Camera source configuration
# Change ONE thing to switch sources:
# - `.env` file variable (recommended): CAMERA1_SOURCE / CAMERA2_SOURCE / CAMERA3_SOURCE
# - or environment variable: same names
# Examples:
# - CAMERA1_SOURCE=webcam        -> uses local webcam index 0
# - CAMERA1_SOURCE=1             -> uses local webcam index 1
# - CAMERA1_SOURCE=rtsp://...    -> uses IP camera RTSP
def _load_env_file(path: str = ".env"):
    """Minimal .env loader (so you don't need python-dotenv)."""
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # For camera switching, `.env` should win (so you only edit one file).
                # For all other vars, don't override real environment variables.
                if key.startswith("CAMERA") and key.endswith("_SOURCE"):
                    os.environ[key] = value
                else:
                    os.environ.setdefault(key, value)
    except Exception as e:
        print(f"⚠️ Could not load .env file '{path}': {e}")


def _parse_camera_source(value: str):
    v = (value or "").strip()
    if not v:
        return None
    if v.lower() in {"webcam", "laptop", "local"}:
        return 0  # Laptop webcam (index 0)
    if v.isdigit():
        return int(v)
    return v

# Load .env before reading camera source variables
_load_env_file(".env")

# Per-camera sources (edit ONLY .env to switch)
DEFAULT_CAMERA1 = "rtsp://admin:Dammah24@172.20.10.12:554/Streaming/Channels/101"  # Note: Capital S in Streaming
CAMERA_SOURCES = {
    1: _parse_camera_source(os.getenv("CAMERA1_SOURCE", DEFAULT_CAMERA1)),
    2: _parse_camera_source(os.getenv("CAMERA2_SOURCE", "0")),
    3: _parse_camera_source(os.getenv("CAMERA3_SOURCE", "0")),
}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = Database()

# Detection service instance (will be initialized after DB)
detection_service = None

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Authentication
security = HTTPBearer(auto_error=False)  # Don't auto-raise error for stream endpoint

# Pydantic Models for Auth
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
    password: str
    full_name: str

# Authentication Dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    user = await db.get_user_by_id(int(user_id))
    if user is None or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user

# API Routes - Authentication

@app.get("/")
async def root():
    return {"message": "INFOCAM Detection API", "status": "running"}

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/login")
async def login(login_data: LoginRequest):
    """Login endpoint - returns JWT token"""
    # Get user from database
    user = await db.get_user_by_email(login_data.email)
    if not user:
        print(f"❌ Login attempt: User not found - {login_data.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Check if user is active
    if not user.get("is_active"):
        print(f"❌ Login attempt: Account deactivated - {login_data.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Debug: Print password verification details
    print(f"🔍 Login attempt for: {login_data.email}")
    print(f"   Stored hash: {user['password_hash'][:20]}...")
    print(f"   Password length: {len(login_data.password)}")
    
    # Verify password
    password_valid = verify_password(login_data.password, user["password_hash"])
    print(f"   Password valid: {password_valid}")
    
    if not password_valid:
        print(f"❌ Login failed: Invalid password for {login_data.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Create access token
    if login_data.remember_me:
        expires_delta = timedelta(days=REMEMBER_ME_EXPIRE_DAYS)
    else:
        expires_delta = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    access_token = create_access_token(
        data={"sub": str(user["id"]), "email": user["email"]},
        expires_delta=expires_delta
    )
    
    # Update last login
    await db.update_last_login(user["id"])
    
    return {
        "token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"]
        },
        "expires_in": int(expires_delta.total_seconds())
    }

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout endpoint (client-side token removal)"""
    return {"message": "Logged out successfully"}

@app.get("/api/auth/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "role": current_user["role"]
    }

@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Request password reset"""
    user = await db.get_user_by_email(request.email)
    if not user:
        # Don't reveal if email exists (security)
        return {"message": "If the email exists, a reset link has been sent"}
    
    # Generate reset token
    reset_token = generate_reset_token()
    expires_at = datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    
    # Store reset token in database
    await db.set_reset_token(request.email, reset_token, expires_at)
    
    # In production, send email here. For now, return token for testing
    # TODO: Send email with reset link: /reset-password?token={reset_token}
    return {
        "message": "Password reset token generated",
        "reset_token": reset_token,  # Remove this in production, send via email
        "expires_in": 3600  # 1 hour
    }

@app.post("/api/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset password using reset token"""
    # Get user by reset token
    user = await db.get_user_by_reset_token(request.token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Validate password strength (min 6 characters)
    if len(request.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    # Hash new password
    password_hash = get_password_hash(request.new_password)
    
    # Update password
    await db.update_user_password(user["id"], password_hash)
    
    return {"message": "Password reset successfully"}

# ==================== ADMIN MANAGEMENT ENDPOINTS ====================

@app.get("/api/admin/users")
async def list_users(current_user: dict = Depends(get_current_user)):
    """List all users (admin only)"""
    users = await db.get_all_users()
    return {"users": users}

@app.post("/api/admin/users")
async def add_admin(request: AddAdminRequest, current_user: dict = Depends(get_current_user)):
    """Add a new admin"""
    # Check if email already exists
    existing_user = await db.get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    
    # Validate password strength
    if len(request.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    # Hash password
    password_hash = get_password_hash(request.password)
    
    # Create user
    user_id = await db.create_user(request.email, password_hash, request.full_name, "admin")
    
    return {
        "message": "Admin created successfully",
        "user_id": user_id
    }

@app.delete("/api/admin/users/{user_id}")
async def remove_admin(user_id: int, current_user: dict = Depends(get_current_user)):
    """Remove an admin"""
    # Prevent self-deletion
    if user_id == current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    # Check if user exists
    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Delete user
    success = await db.delete_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )
    
    return {"message": "Admin removed successfully"}

@app.get("/api/cameras")
async def get_cameras(current_user: dict = Depends(get_current_user)):
    """Get list of available cameras"""
    cameras = [
        {"id": 1, "name": "Camera 01", "source": str(CAMERA_SOURCES.get(1)), "status": "available"},
        {"id": 2, "name": "Camera 02", "source": str(CAMERA_SOURCES.get(2)), "status": "available"},
        {"id": 3, "name": "Camera 03", "source": str(CAMERA_SOURCES.get(3)), "status": "available"},
    ]
    return {"cameras": cameras}

@app.post("/api/cameras/{camera_id}/start")
async def start_detection(camera_id: int, current_user: dict = Depends(get_current_user)):
    """Start detection for a camera"""
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    try:
        # Camera source selection:
        # - Uses CAMERA_SOURCES configured from `.env` / env vars
        # - Defaults to webcam 0 if not configured
        camera_source = CAMERA_SOURCES.get(camera_id, 1)
        
        camera_info = {"id": camera_id, "source": camera_source}
        success = await detection_service.start_detection(camera_id, camera_info, manager)
        if success:
            return {"message": f"Detection started for camera {camera_id}", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Detection already running or camera not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cameras/{camera_id}/stop")
async def stop_detection(camera_id: int, current_user: dict = Depends(get_current_user)):
    """Stop detection for a camera"""
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    try:
        success = await detection_service.stop_detection(camera_id)
        if success:
            return {"message": f"Detection stopped for camera {camera_id}", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Detection not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras/{camera_id}/status")
async def get_detection_status(camera_id: int, current_user: dict = Depends(get_current_user)):
    """Get detection status for a camera"""
    if detection_service is None:
        return {"camera_id": camera_id, "is_running": False}
    is_running = detection_service.is_detection_running(camera_id)
    return {"camera_id": camera_id, "is_running": is_running}

@app.get("/api/cameras/{camera_id}/stream")
async def video_stream(
    camera_id: int, 
    token: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get live video stream from camera"""
    # Validate authentication - either from header or query parameter (for img tag compatibility)
    authenticated = False
    
    # Try header first (normal API calls)
    if credentials:
        payload = verify_token(credentials.credentials)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                user = await db.get_user_by_id(int(user_id))
                if user and user.get("is_active"):
                    authenticated = True
    
    # Try query parameter (for img tag)
    if not authenticated and token:
        payload = verify_token(token)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                user = await db.get_user_by_id(int(user_id))
                if user and user.get("is_active"):
                    authenticated = True
    
    if not authenticated:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if detection_service is None:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    return StreamingResponse(
        detection_service.get_video_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: int):
    # WebSocket doesn't use HTTPBearer, but we can validate token from query params if needed
    """WebSocket endpoint for real-time detection alerts"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            await websocket.send_json({"type": "ping", "message": "connected"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/violations")
async def get_violations(
    camera_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get violations with optional filters"""
    violations = await db.get_violations(
        camera_id=camera_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    return {"violations": violations, "count": len(violations)}

@app.get("/api/violations/stats")
async def get_violation_stats(
    period: str = "month",  # "month" or "week"
    camera_id: Optional[int] = None,
    detection_class: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get violation statistics grouped by period"""
    stats = await db.get_violation_stats(period=period, camera_id=camera_id, detection_class=detection_class)
    return {"stats": stats, "period": period}

@app.get("/api/violations/{violation_id}/video")
async def download_violation_video(violation_id: int, current_user: dict = Depends(get_current_user)):
    """Download video clip for a violation"""
    violation = await db.get_violation_by_id(violation_id)
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    video_path = violation["video_path"]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """Get dashboard statistics"""
    stats = await db.get_dashboard_stats()
    return stats

@app.on_event("startup")
async def startup_event():
    """Initialize database and detection service on startup"""
    await db.init_db()
    global detection_service
    detection_service = DetectionService(db)
    print("✅ Application startup complete - Database initialized")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


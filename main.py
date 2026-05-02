from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
import uvicorn
import asyncio
from datetime import datetime, timedelta
import os

from app.detection_service import DetectionService
from app.database import Database
from app.models import ViolationCreate, ViolationResponse, LoginRequest, ForgotPasswordRequest, ResetPasswordRequest, AddAdminRequest, SetPasswordRequest, GoogleAuthRequest
from app.auth import verify_password, get_password_hash, create_access_token, verify_token, generate_reset_token, verify_google_token, send_invite_email, ACCESS_TOKEN_EXPIRE_HOURS, REMEMBER_ME_EXPIRE_DAYS


def _load_env(path: str = ".env"):
    try:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if key.startswith("CAMERA") and key.endswith("_SOURCE"):
                    os.environ[key] = value
                else:
                    os.environ.setdefault(key, value)
    except Exception as e:
        print(f"Could not load .env: {e}")


def _parse_camera_source(value: str):
    v = (value or "").strip()
    if not v:
        return None
    if v.lower() in {"webcam", "laptop", "local"}:
        return 0
    if v.isdigit():
        return int(v)
    return v


_load_env(".env")

DEFAULT_CAMERA1 = "rtsp://admin:Dammah24@172.20.10.12:554/Streaming/Channels/101"
CAMERA_SOURCES  = {
    1: _parse_camera_source(os.getenv("CAMERA1_SOURCE", DEFAULT_CAMERA1)),
    2: _parse_camera_source(os.getenv("CAMERA2_SOURCE", "0")),
    3: _parse_camera_source(os.getenv("CAMERA3_SOURCE", "0")),
}

db                = Database()
detection_service = None
security          = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    global detection_service
    detection_service = DetectionService(db)
    print("Application startup complete")
    yield


app = FastAPI(title="INFOCAM Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# ── Auth dependency ─────────────────────────────────────────────────────

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token payload")
    user = await db.get_user_by_id(int(user_id))
    if user is None or not user.get("is_active"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user


async def _auth_from_request(credentials: Optional[HTTPAuthorizationCredentials], token: Optional[str]) -> bool:
    """Validate auth from header or query-param token (for img-tag stream compatibility)."""
    for t in filter(None, [credentials.credentials if credentials else None, token]):
        payload = verify_token(t)
        if payload and payload.get("sub"):
            user = await db.get_user_by_id(int(payload["sub"]))
            if user and user.get("is_active"):
                return True
    return False


# ── Auth endpoints ──────────────────────────────────────────────────────

@app.get("/api/status")
async def root():
    return {"message": "INFOCAM Detection API", "status": "running"}


@app.get("/api/config")
async def get_config():
    return {"google_client_id": os.getenv("GOOGLE_CLIENT_ID", "")}


@app.post("/api/auth/login")
async def login(data: LoginRequest):
    user = await db.get_user_by_email(data.email)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if not user.get("is_active"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is deactivated")
    if user.get("auth_provider") == "google":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This account uses Google Sign-In. Please use the Google button.")
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")

    expires = timedelta(days=REMEMBER_ME_EXPIRE_DAYS) if data.remember_me else timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    token   = create_access_token({"sub": str(user["id"]), "email": user["email"]}, expires)
    await db.update_last_login(user["id"])
    return {
        "token": token, "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "full_name": user["full_name"], "role": user["role"]},
        "expires_in": int(expires.total_seconds()),
    }


@app.post("/api/auth/google")
async def google_login(req: GoogleAuthRequest):
    idinfo = verify_google_token(req.credential)
    if not idinfo:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Google credential")
    email     = idinfo["email"]
    full_name = idinfo.get("name", email.split("@")[0])
    user = await db.get_user_by_email(email)
    if not user:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This Google account has not been invited. Contact your admin.")
    if not user.get("is_active"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is deactivated")
    # Link Google to an invited email/password account on first Google sign-in
    if user.get("auth_provider") == "local":
        await db.set_auth_provider(user["id"], "google")
    await db.update_last_login(user["id"])
    expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    token   = create_access_token({"sub": str(user["id"]), "email": user["email"]}, expires)
    return {
        "token": token, "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "full_name": user["full_name"], "role": user["role"]},
        "expires_in": int(expires.total_seconds()),
    }


@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    return {"message": "Logged out successfully"}


@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return {k: current_user[k] for k in ("id", "email", "full_name", "role")}


@app.post("/api/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    user = await db.get_user_by_email(req.email)
    if not user:
        return {"message": "If the email exists, a reset link has been sent"}
    reset_token = generate_reset_token()
    expires_at  = datetime.utcnow() + timedelta(hours=1)
    await db.set_reset_token(req.email, reset_token, expires_at)
    return {"message": "Password reset token generated", "reset_token": reset_token, "expires_in": 3600}


@app.post("/api/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    user = await db.get_user_by_reset_token(req.token)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired reset token")
    if len(req.new_password) < 6:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password must be at least 6 characters long")
    await db.update_user_password(user["id"], get_password_hash(req.new_password))
    return {"message": "Password reset successfully"}


# ── Admin endpoints ─────────────────────────────────────────────────────

@app.get("/api/admin/users")
async def list_users(current_user: dict = Depends(get_current_user)):
    return {"users": await db.get_all_users()}


@app.post("/api/admin/users")
async def add_admin(req: AddAdminRequest, request: Request, current_user: dict = Depends(get_current_user)):
    if await db.get_user_by_email(req.email):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")
    user_id     = await db.create_user(req.email, "", req.full_name, "admin", "local", is_active=False, email_verified=False)
    invite_token = generate_reset_token()
    expires_at   = datetime.utcnow() + timedelta(hours=24)
    await db.set_invite_token(user_id, invite_token, expires_at)
    base_url     = str(request.base_url).rstrip("/")
    invite_link  = f"{base_url}/?invite={invite_token}"
    sent         = send_invite_email(req.email, req.full_name, invite_link)
    return {
        "message": "Invitation sent" if sent else "User created (email not configured — share link manually)",
        "user_id": user_id,
        "invite_link": invite_link,
        "email_sent": sent,
    }


@app.get("/api/auth/invite/{token}")
async def check_invite(token: str):
    user = await db.get_user_by_invite_token(token)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired invite link")
    return {"valid": True, "email": user["email"], "full_name": user["full_name"]}


@app.post("/api/auth/invite")
async def accept_invite(req: SetPasswordRequest):
    user = await db.get_user_by_invite_token(req.token)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid or expired invite link")
    if len(req.password) < 6:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password must be at least 6 characters")
    await db.activate_invited_user(user["id"], get_password_hash(req.password))
    expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    token   = create_access_token({"sub": str(user["id"]), "email": user["email"]}, expires)
    return {
        "token": token, "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "full_name": user["full_name"], "role": user["role"]},
    }


@app.patch("/api/admin/users/{user_id}/toggle")
async def toggle_admin(user_id: int, current_user: dict = Depends(get_current_user)):
    if user_id == current_user["id"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot deactivate your own account")
    result = await db.toggle_user_active(user_id)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    state = "activated" if result["is_active"] else "deactivated"
    return {"message": f"User {state}", "is_active": result["is_active"]}


@app.delete("/api/admin/users/{user_id}")
async def remove_admin(user_id: int, current_user: dict = Depends(get_current_user)):
    if user_id == current_user["id"]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Cannot delete your own account")
    if not await db.get_user_by_id(user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if not await db.delete_user(user_id):
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to delete user")
    return {"message": "Admin removed successfully"}


# ── Camera endpoints ────────────────────────────────────────────────────

@app.get("/api/cameras")
async def get_cameras(current_user: dict = Depends(get_current_user)):
    cameras = [
        {"id": i, "name": f"Camera 0{i}", "source": str(CAMERA_SOURCES.get(i)), "status": "available"}
        for i in (1, 2, 3)
    ]
    return {"cameras": cameras}


@app.post("/api/cameras/{camera_id}/start")
async def start_detection(camera_id: int, current_user: dict = Depends(get_current_user)):
    if detection_service is None:
        raise HTTPException(503, "Detection service not initialized")
    source = CAMERA_SOURCES.get(camera_id, 1)
    success = await detection_service.start_detection(camera_id, {"id": camera_id, "source": source}, manager)
    if not success:
        raise HTTPException(400, "Detection already running or camera not available")
    return {"message": f"Detection started for camera {camera_id}", "status": "success"}


@app.post("/api/cameras/{camera_id}/stop")
async def stop_detection(camera_id: int, current_user: dict = Depends(get_current_user)):
    if detection_service is None:
        raise HTTPException(503, "Detection service not initialized")
    if not await detection_service.stop_detection(camera_id):
        raise HTTPException(400, "Detection not running")
    return {"message": f"Detection stopped for camera {camera_id}", "status": "success"}


@app.get("/api/cameras/{camera_id}/status")
async def get_status(camera_id: int, current_user: dict = Depends(get_current_user)):
    is_running = detection_service.is_detection_running(camera_id) if detection_service else False
    return {"camera_id": camera_id, "is_running": is_running}


@app.get("/api/cameras/{camera_id}/stream")
async def video_stream(
    camera_id: int,
    token: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if not await _auth_from_request(credentials, token):
        raise HTTPException(401, "Not authenticated")
    if detection_service is None:
        raise HTTPException(503, "Detection service not initialized")
    return StreamingResponse(
        detection_service.get_video_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: int):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json({"type": "ping", "message": "connected"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Violation endpoints ─────────────────────────────────────────────────

@app.get("/api/violations")
async def get_violations(
    camera_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user),
):
    violations = await db.get_violations(camera_id=camera_id, start_date=start_date, end_date=end_date, limit=limit)
    for v in violations:
        v["has_video"] = bool(v.get("video_path") and os.path.exists(v["video_path"]))
    return {"violations": violations, "count": len(violations)}


@app.get("/api/violations/stats")
async def get_violation_stats(
    period: str = "month",
    camera_id: Optional[int] = None,
    detection_class: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    stats = await db.get_violation_stats(period=period, camera_id=camera_id, detection_class=detection_class)
    return {"stats": stats, "period": period}


@app.get("/api/violations/{violation_id}/video")
async def download_video(violation_id: int, current_user: dict = Depends(get_current_user)):
    violation = await db.get_violation_by_id(violation_id)
    if not violation:
        raise HTTPException(404, "Violation not found")
    path = violation["video_path"]
    if not os.path.exists(path):
        raise HTTPException(404, "Video file not found")
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))


@app.get("/api/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    return await db.get_dashboard_stats()


app.mount("/", StaticFiles(directory="FYP-UI-REPO", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import bcrypt
import secrets
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production-min-32-chars")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24  # Default: 24 hours
REMEMBER_ME_EXPIRE_DAYS = 7     # Remember me: 7 days

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        # hashed_password is stored as string, need to encode it
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    """Hash a password"""
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    # Return as string for database storage
    return hashed.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)


def send_invite_email(to_email: str, full_name: str, invite_link: str) -> bool:
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_EMAIL", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()
    if not smtp_user or not smtp_pass:
        print(f"[INVITE] SMTP not configured — link: {invite_link}")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "You've been invited to INFOCAM"
    msg["From"]    = f"INFOCAM <{smtp_user}>"
    msg["To"]      = to_email
    html = f"""
    <div style="font-family:Inter,sans-serif;max-width:500px;margin:auto;padding:32px;">
      <h2 style="color:#1e293b;">You've been invited to INFOCAM</h2>
      <p style="color:#475569;">Hi {full_name}, an admin has invited you to access the INFOCAM violation detection system.</p>
      <p style="color:#475569;">Click the button below to verify your email and set your password:</p>
      <a href="{invite_link}" style="display:inline-block;margin:20px 0;padding:12px 28px;background:#2563eb;color:#fff;border-radius:8px;text-decoration:none;font-weight:600;">
        Accept Invitation
      </a>
      <p style="color:#94a3b8;font-size:12px;">This link expires in 24 hours. If you didn't expect this, ignore this email.</p>
    </div>"""
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[INVITE] Email failed: {e}")
        return False


def verify_google_token(credential: str) -> Optional[dict]:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        return None
    try:
        return id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            client_id,
        )
    except Exception:
        return None


# INFOCAM — Violation Detection System

AI-powered smoking violation detection using YOLO. Detects **Person + Cigarette + Smoke** in real time via RTSP or local webcam, logs violations to a PostgreSQL database, saves video clips, and streams live to a web dashboard.

## Features

- Real-time detection of Person, Cigarette, and Smoke using dual YOLO models
- Live MJPEG camera stream with bounding box overlay
- Multi-camera support (up to 3 cameras — RTSP or webcam)
- Automatic 15-second video clip recording on violation (5 s pre + 10 s post)
- Violations log with filters and per-violation clip download (shows "No Clip" when recording was interrupted)
- Dashboard with charts (monthly, weekly, 24-hour summary)
- Google OAuth login + local email/password login
- Admin user management: invite by email, deactivate/reactivate, auth provider badges
- Email invite flow — inactive account created → invite link sent via Gmail SMTP → user sets password → account activated
- PostgreSQL storage with IP geolocation tagging

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment** — create a `.env` file:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=infocam_db
   DB_USER=postgres
   DB_PASSWORD=yourpassword

   GOOGLE_CLIENT_ID=your_google_oauth_client_id
   GOOGLE_MAPS_API_KEY=your_maps_api_key

   SMTP_EMAIL=you@gmail.com
   SMTP_PASSWORD=your_gmail_app_password

   CAMERA1_SOURCE=rtsp://user:pass@ip:554/stream
   # CAMERA1_SOURCE=webcam   ← use this for local webcam
   ```

   > **Gmail App Password**: Go to myaccount.google.com → Security → 2-Step Verification → App passwords. Do NOT use your real Gmail password.

3. **Run the server:**
   ```bash
   python main.py
   ```

4. **Open the UI:** http://localhost:8000

   API docs: http://localhost:8000/docs

5. **First admin account:** Use the database directly or the `/api/admin/users` endpoint to create the initial admin. All subsequent users are invited via the dashboard.

## Authentication

- **Google Sign-In**: Only users already added by an admin can sign in with Google. Unknown Google accounts are rejected.
- **Email/Password**: Used for accounts created via the invite flow.
- **Invite flow**: Admin adds a user (name + email) → server creates an inactive account and sends an invite email → user clicks the link, sets a password → account is activated.
- Accounts are tied to one provider. Google accounts cannot log in with email/password and vice versa.

## Project Structure

```
├── main.py                  # FastAPI app & all API routes
├── app/
│   ├── detection_service.py # YOLO inference, clip recording, violation logic
│   ├── database.py          # PostgreSQL queries (asyncpg)
│   ├── auth.py              # JWT tokens, Google OAuth verification, invite emails
│   └── models.py            # Pydantic request/response models
├── FYP-UI-REPO/             # Frontend (vanilla JS, Chart.js) — served at /
├── weights/                 # YOLO model weights (not tracked in git)
│   ├── best.pt              # Main model (person + cigarette)
│   └── 11k.pt               # Smoke model
└── clips/                   # Saved violation video clips
```

## Detection Logic

A violation is triggered when all three are detected together with spatial proximity, and a 20-second cooldown has elapsed:

1. Main model detects **Person** (≥ 0.70 conf) and **Cigarette** (≥ 0.50 conf) — spatially matched
2. Smoke model (conf ≥ 0.20 — kept low because smoke is translucent) detects **Smoke** near the combined person+cigarette bounding box with a 30% margin and up to 1.5× bbox distance
3. Smoke detections are buffered for 2 seconds to bridge gaps between frames
4. A 15-second clip is saved to `clips/`
5. Violation is stored in PostgreSQL with location, confidence, and bounding box data
6. A WebSocket alert is broadcast to all connected dashboard clients

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `DB_HOST` | Yes | PostgreSQL host |
| `DB_PORT` | Yes | PostgreSQL port (default 5432) |
| `DB_NAME` | Yes | Database name |
| `DB_USER` | Yes | Database user |
| `DB_PASSWORD` | Yes | Database password |
| `GOOGLE_CLIENT_ID` | Yes | OAuth 2.0 Client ID from Google Cloud Console |
| `GOOGLE_MAPS_API_KEY` | Yes | Maps API key for location display |
| `SMTP_EMAIL` | Yes | Gmail address for sending invite emails |
| `SMTP_PASSWORD` | Yes | Gmail App Password (not your login password) |
| `CAMERA1_SOURCE` | Yes | RTSP URL or `webcam` |
| `CAMERA2_SOURCE` | No | Second camera (RTSP URL or `0`) |
| `CAMERA3_SOURCE` | No | Third camera |

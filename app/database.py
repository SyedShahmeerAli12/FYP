import asyncpg
import os
from datetime import datetime
from typing import Optional, List, Dict
import json


class Database:
    def __init__(self):
        self.db_host     = os.getenv("DB_HOST", "localhost")
        self.db_port     = int(os.getenv("DB_PORT", 5432))
        self.db_name     = os.getenv("DB_NAME", "infocam_db")
        self.db_user     = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "12345")
        self.pool        = None

    async def get_pool(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.db_host, port=self.db_port,
                database=self.db_name, user=self.db_user, password=self.db_password,
                min_size=1, max_size=10,
            )
        return self.pool

    async def init_db(self):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    location VARCHAR(200),
                    ip_address VARCHAR(50),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("""
                INSERT INTO cameras (id, name, location, ip_address)
                VALUES (1, 'Camera 01', 'Home', NULL)
                ON CONFLICT (id) DO NOTHING
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id SERIAL PRIMARY KEY,
                    camera_id INTEGER NOT NULL,
                    detection_class VARCHAR(100) NOT NULL,
                    confidence FLOAT NOT NULL,
                    video_path VARCHAR(500) NOT NULL,
                    priority VARCHAR(20),
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    frame_count INTEGER,
                    bbox_data JSONB,
                    location VARCHAR(200),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute("ALTER TABLE violations ADD COLUMN IF NOT EXISTS priority VARCHAR(20)")
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_camera_id ON violations(camera_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_location ON violations(location)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_created_at ON violations(created_at DESC)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_priority ON violations(priority)
            """)
            await conn.execute("""
                UPDATE violations
                SET priority = CASE
                    WHEN confidence >= 0.85 THEN 'high'
                    WHEN confidence >= 0.70 THEN 'medium'
                    ELSE 'low'
                END
                WHERE priority IS NULL
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    role VARCHAR(50) DEFAULT 'admin',
                    is_active BOOLEAN DEFAULT TRUE,
                    reset_token VARCHAR(255) NULL,
                    reset_token_expires TIMESTAMP NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP NULL
                )
            """)
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider VARCHAR(20) DEFAULT 'local'")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT TRUE")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS invite_token VARCHAR(255)")
            await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS invite_token_expires TIMESTAMP")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_invite_token ON users(invite_token)")
            print("Database initialized successfully")

    @staticmethod
    def _priority(confidence: float) -> str:
        if confidence >= 0.85: return "high"
        if confidence >= 0.70: return "medium"
        return "low"

    @staticmethod
    def _user_dict(row) -> Dict:
        return {
            "id": row["id"], "email": row["email"],
            "password_hash": row["password_hash"], "full_name": row["full_name"],
            "role": row["role"], "is_active": row["is_active"],
            "auth_provider": row["auth_provider"] if "auth_provider" in row.keys() else "local",
            "last_login": row["last_login"],
        }

    # ── Violations ──────────────────────────────────────────────────────

    async def create_violation(self, data: Dict) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            confidence = float(data["confidence"])
            priority   = data.get("priority") or self._priority(confidence)
            return await conn.fetchval("""
                INSERT INTO violations
                (camera_id, detection_class, confidence, video_path, priority, timestamp, frame_count, bbox_data)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id
            """,
                data["camera_id"], data["detection_class"], confidence,
                data["video_path"], priority, data["timestamp"],
                data.get("frame_count", 0), json.dumps(data.get("bbox_data", {})),
            )

    async def get_violations(
        self,
        camera_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            q, params, n = "SELECT * FROM violations WHERE 1=1", [], 0
            if camera_id:
                n += 1; q += f" AND camera_id = ${n}"; params.append(camera_id)
            if start_date:
                n += 1; q += f" AND timestamp >= ${n}::timestamp"; params.append(start_date)
            if end_date:
                n += 1; q += f" AND timestamp <= ${n}::timestamp"; params.append(end_date)
            q += f" ORDER BY timestamp DESC LIMIT ${n+1}"; params.append(limit)
            rows = await conn.fetch(q, *params)
            return [self._violation_dict(r) for r in rows]

    async def get_violation_by_id(self, violation_id: int) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM violations WHERE id = $1", violation_id)
            return self._violation_dict(row) if row else None

    def _violation_dict(self, row) -> Dict:
        conf = float(row["confidence"])
        return {
            "id": row["id"], "camera_id": row["camera_id"],
            "detection_class": row["detection_class"], "confidence": conf,
            "video_path": row["video_path"],
            "priority": row["priority"] or self._priority(conf),
            "timestamp": row["timestamp"].isoformat(),
            "frame_count": row["frame_count"],
            "bbox_data": row["bbox_data"] if row["bbox_data"] else {},
            "created_at": row["created_at"].isoformat(),
        }

    async def get_violation_stats(
        self,
        period: str = "month",
        camera_id: Optional[int] = None,
        detection_class: Optional[str] = None,
    ) -> List[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            trunc = "month" if period == "month" else "week"
            group = f"DATE_TRUNC('{trunc}', timestamp)"
            q = f"""
                SELECT {group} as period, COUNT(*) as count,
                       AVG(confidence) as avg_confidence, MAX(confidence) as max_confidence
                FROM violations WHERE 1=1
            """
            params, n = [], 0
            if camera_id:
                n += 1; q += f" AND camera_id = ${n}"; params.append(camera_id)
            if detection_class and detection_class.strip():
                n += 1; q += f" AND detection_class = ${n}"; params.append(detection_class.strip())
            q += f" GROUP BY {group} ORDER BY period DESC LIMIT 7"
            rows = await conn.fetch(q, *params)
            return [{
                "period": str(r["period"]), "count": r["count"],
                "avg_confidence": float(r["avg_confidence"]) if r["avg_confidence"] else 0,
                "max_confidence": float(r["max_confidence"]) if r["max_confidence"] else 0,
            } for r in rows]

    # ── Users ───────────────────────────────────────────────────────────

    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id,email,password_hash,full_name,role,is_active,auth_provider,last_login FROM users WHERE email=$1",
                email)
            return self._user_dict(row) if row else None

    async def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id,email,password_hash,full_name,role,is_active,auth_provider,last_login FROM users WHERE id=$1",
                user_id)
            return self._user_dict(row) if row else None

    async def get_user_by_reset_token(self, reset_token: str) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id,email,password_hash,full_name,role,is_active,auth_provider,last_login
                FROM users WHERE reset_token=$1 AND reset_token_expires > NOW()
            """, reset_token)
            return self._user_dict(row) if row else None

    async def create_user(self, email: str, password_hash: str, full_name: str, role: str = "admin", auth_provider: str = "local", is_active: bool = True, email_verified: bool = True) -> int:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "INSERT INTO users (email,password_hash,full_name,role,auth_provider,is_active,email_verified) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id",
                email, password_hash, full_name, role, auth_provider, is_active, email_verified)

    async def set_invite_token(self, user_id: int, token: str, expires_at: datetime):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET invite_token=$1, invite_token_expires=$2 WHERE id=$3",
                token, expires_at, user_id)

    async def get_user_by_invite_token(self, token: str) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id,email,password_hash,full_name,role,is_active,auth_provider,last_login
                FROM users WHERE invite_token=$1 AND invite_token_expires > NOW()
            """, token)
            return self._user_dict(row) if row else None

    async def set_auth_provider(self, user_id: int, provider: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE users SET auth_provider=$1 WHERE id=$2", provider, user_id)

    async def activate_invited_user(self, user_id: int, password_hash: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET password_hash=$1, is_active=TRUE, email_verified=TRUE,
                invite_token=NULL, invite_token_expires=NULL WHERE id=$2
            """, password_hash, user_id)

    async def update_user_password(self, user_id: int, password_hash: str):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET password_hash=$1, reset_token=NULL, reset_token_expires=NULL WHERE id=$2
            """, password_hash, user_id)

    async def set_reset_token(self, email: str, reset_token: str, expires_at: datetime):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET reset_token=$1, reset_token_expires=$2 WHERE email=$3",
                reset_token, expires_at, email)

    async def update_last_login(self, user_id: int):
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("UPDATE users SET last_login=NOW() WHERE id=$1", user_id)

    async def get_all_users(self) -> List[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id,email,full_name,role,is_active,auth_provider,created_at,last_login FROM users ORDER BY created_at DESC")
            return [{
                "id": r["id"], "email": r["email"], "full_name": r["full_name"],
                "role": r["role"], "is_active": r["is_active"],
                "auth_provider": r["auth_provider"] or "local",
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "last_login": r["last_login"].isoformat() if r["last_login"] else None,
            } for r in rows]

    async def toggle_user_active(self, user_id: int) -> Optional[Dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE users SET is_active = NOT is_active WHERE id=$1 RETURNING id,is_active", user_id)
            return {"id": row["id"], "is_active": row["is_active"]} if row else None

    async def delete_user(self, user_id: int) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM users WHERE id=$1", user_id)
            return result == "DELETE 1"

    # ── Dashboard ───────────────────────────────────────────────────────

    async def get_dashboard_stats(self) -> Dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # All counts in a single round-trip
            counts = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE DATE(timestamp) = CURRENT_DATE) as today,
                    COUNT(*) FILTER (WHERE timestamp >= DATE_TRUNC('week',  CURRENT_TIMESTAMP)) as week,
                    COUNT(*) FILTER (WHERE timestamp >= DATE_TRUNC('month', CURRENT_TIMESTAMP)) as month,
                    COUNT(*) FILTER (WHERE COALESCE(priority,
                        CASE WHEN confidence>=0.85 THEN 'high' WHEN confidence>=0.70 THEN 'medium' ELSE 'low' END
                    ) = 'high') as high_priority,
                    COUNT(*) FILTER (WHERE COALESCE(priority,
                        CASE WHEN confidence>=0.85 THEN 'high' WHEN confidence>=0.70 THEN 'medium' ELSE 'low' END
                    ) = 'medium') as medium_priority,
                    COUNT(*) FILTER (WHERE COALESCE(priority,
                        CASE WHEN confidence>=0.85 THEN 'high' WHEN confidence>=0.70 THEN 'medium' ELSE 'low' END
                    ) = 'low') as low_priority
                FROM violations
            """)

            top_class = await conn.fetchrow("""
                SELECT detection_class, COUNT(*) as count
                FROM violations GROUP BY detection_class ORDER BY count DESC LIMIT 1
            """)
            top_violations = await conn.fetch("""
                SELECT DATE(timestamp) as violation_date, COUNT(*) as count
                FROM violations GROUP BY DATE(timestamp) ORDER BY violation_date DESC LIMIT 7
            """)
            top_outlets = await conn.fetch("""
                SELECT COALESCE(location,'Unknown') as location, COUNT(*) as count
                FROM violations GROUP BY location ORDER BY count DESC LIMIT 7
            """)
            detection_classes = await conn.fetch("""
                SELECT DISTINCT detection_class FROM violations
                WHERE detection_class IS NOT NULL ORDER BY detection_class
            """)
            hourly = await conn.fetch("""
                SELECT DATE_TRUNC('hour', timestamp) as hour, COUNT(*) as count
                FROM violations WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY DATE_TRUNC('hour', timestamp) ORDER BY hour ASC
            """)

        return {
            "total_violations":   counts["total"]  or 0,
            "today_violations":   counts["today"]  or 0,
            "week_violations":    counts["week"]   or 0,
            "month_violations":   counts["month"]  or 0,
            "high_priority":      counts["high_priority"]   or 0,
            "medium_priority":    counts["medium_priority"] or 0,
            "low_priority":       counts["low_priority"]    or 0,
            "top_violation": {
                "class": top_class["detection_class"] if top_class else "No Data",
                "count": top_class["count"] if top_class else 0,
            },
            "top_violations": [
                {"date": r["violation_date"].isoformat() if hasattr(r["violation_date"], 'isoformat') else str(r["violation_date"]),
                 "count": r["count"]}
                for r in top_violations
            ],
            "detection_classes": [r["detection_class"] for r in detection_classes],
            "top_outlets": [{"location": r["location"], "count": r["count"]} for r in top_outlets],
            "hourly_summary": [
                {"hour": r["hour"].isoformat() if hasattr(r["hour"], 'isoformat') else str(r["hour"]),
                 "count": r["count"]}
                for r in hourly
            ],
        }

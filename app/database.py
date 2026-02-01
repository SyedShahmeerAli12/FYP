import asyncpg
import os
from datetime import datetime
from typing import Optional, List, Dict
import json

class Database:
    def __init__(self):
        # PostgreSQL connection settings
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", 5432))
        self.db_name = os.getenv("DB_NAME", "infocam_db")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "12345")  # Default password
        self.pool = None
    
    async def get_pool(self):
        """Get or create connection pool"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                min_size=1,
                max_size=10
            )
        return self.pool
    
    async def init_db(self):
        """Initialize database tables"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Create cameras table (for location management)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    location VARCHAR(200),
                    ip_address VARCHAR(50),
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default camera if not exists
            await conn.execute("""
                INSERT INTO cameras (id, name, location, ip_address)
                VALUES (1, 'Camera 01', 'Home', NULL)
                ON CONFLICT (id) DO NOTHING
            """)
            
            # Create violations table with location
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

            # If table already existed from older versions, ensure priority column exists
            await conn.execute("""
                ALTER TABLE violations
                ADD COLUMN IF NOT EXISTS priority VARCHAR(20)
            """)
            
            # Create indexes for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp 
                ON violations(timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_camera_id 
                ON violations(camera_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_location 
                ON violations(location)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_created_at 
                ON violations(created_at DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_violations_priority
                ON violations(priority)
            """)

            # Backfill priority for existing rows (only where NULL)
            await conn.execute("""
                UPDATE violations
                SET priority = CASE
                    WHEN confidence >= 0.85 THEN 'high'
                    WHEN confidence >= 0.70 THEN 'medium'
                    ELSE 'low'
                END
                WHERE priority IS NULL
            """)
            
            # Create users table for authentication
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
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_email 
                ON users(email)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_reset_token 
                ON users(reset_token)
            """)
            
            print("✅ Database initialized successfully")

    @staticmethod
    def _priority_from_confidence(confidence: float) -> str:
        """Map a confidence score to a priority label."""
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.70:
            return "medium"
        return "low"
    
    async def create_violation(self, violation_data: Dict) -> int:
        """Create a new violation record"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            confidence = float(violation_data["confidence"])
            priority = violation_data.get("priority") or self._priority_from_confidence(confidence)
            violation_id = await conn.fetchval("""
                INSERT INTO violations 
                (camera_id, detection_class, confidence, video_path, priority, timestamp, frame_count, bbox_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """,
                violation_data["camera_id"],
                violation_data["detection_class"],
                confidence,
                violation_data["video_path"],
                priority,
                violation_data["timestamp"],
                violation_data.get("frame_count", 0),
                json.dumps(violation_data.get("bbox_data", {}))
            )
            return violation_id
    
    async def get_violations(
        self,
        camera_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get violations with filters"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            query = "SELECT * FROM violations WHERE 1=1"
            params = []
            param_count = 0
            
            if camera_id:
                param_count += 1
                query += f" AND camera_id = ${param_count}"
                params.append(camera_id)
            
            if start_date:
                param_count += 1
                query += f" AND timestamp >= ${param_count}::timestamp"
                params.append(start_date)
            
            if end_date:
                param_count += 1
                query += f" AND timestamp <= ${param_count}::timestamp"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC LIMIT $" + str(param_count + 1)
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            violations = []
            for row in rows:
                violations.append({
                    "id": row["id"],
                    "camera_id": row["camera_id"],
                    "detection_class": row["detection_class"],
                    "confidence": float(row["confidence"]),
                    "video_path": row["video_path"],
                    "priority": row["priority"] or self._priority_from_confidence(float(row["confidence"])),
                    "timestamp": row["timestamp"].isoformat(),
                    "frame_count": row["frame_count"],
                    "bbox_data": row["bbox_data"] if row["bbox_data"] else {},
                    "created_at": row["created_at"].isoformat()
                })
            
            return violations
    
    async def get_violation_by_id(self, violation_id: int) -> Optional[Dict]:
        """Get a single violation by ID"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM violations WHERE id = $1",
                violation_id
            )
            
            if row:
                return {
                    "id": row["id"],
                    "camera_id": row["camera_id"],
                    "detection_class": row["detection_class"],
                    "confidence": float(row["confidence"]),
                    "video_path": row["video_path"],
                    "priority": row["priority"] or self._priority_from_confidence(float(row["confidence"])),
                    "timestamp": row["timestamp"].isoformat(),
                    "frame_count": row["frame_count"],
                    "bbox_data": row["bbox_data"] if row["bbox_data"] else {},
                    "created_at": row["created_at"].isoformat()
                }
            return None
    
    async def get_violation_stats(
        self,
        period: str = "month",
        camera_id: Optional[int] = None,
        detection_class: Optional[str] = None
    ) -> List[Dict]:
        """Get violation statistics grouped by period"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            if period == "month":
                date_format = "YYYY-MM"
                group_by = "DATE_TRUNC('month', timestamp)"
            else:  # week
                date_format = "YYYY-\"W\"WW"
                group_by = "DATE_TRUNC('week', timestamp)"
            
            query = f"""
                SELECT 
                    {group_by} as period,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    MAX(confidence) as max_confidence
                FROM violations
                WHERE 1=1
            """
            params = []
            param_count = 0
            
            if camera_id:
                param_count += 1
                query += f" AND camera_id = ${param_count}"
                params.append(camera_id)

            if detection_class and detection_class.strip():
                param_count += 1
                query += f" AND detection_class = ${param_count}"
                params.append(detection_class.strip())
            
            query += f" GROUP BY {group_by} ORDER BY period DESC LIMIT 7"
            
            rows = await conn.fetch(query, *params)
            
            stats = []
            for row in rows:
                stats.append({
                    "period": str(row["period"]),
                    "count": row["count"],
                    "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0,
                    "max_confidence": float(row["max_confidence"]) if row["max_confidence"] else 0
                })
            
            return stats
    
    # ==================== AUTHENTICATION METHODS ====================
    
    async def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, password_hash, full_name, role, is_active, last_login
                FROM users WHERE email = $1
            """, email)
            if row:
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "password_hash": row["password_hash"],
                    "full_name": row["full_name"],
                    "role": row["role"],
                    "is_active": row["is_active"],
                    "last_login": row["last_login"]
                }
            return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, password_hash, full_name, role, is_active, last_login
                FROM users WHERE id = $1
            """, user_id)
            if row:
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "password_hash": row["password_hash"],
                    "full_name": row["full_name"],
                    "role": row["role"],
                    "is_active": row["is_active"],
                    "last_login": row["last_login"]
                }
            return None
    
    async def create_user(self, email: str, password_hash: str, full_name: str, role: str = "admin") -> int:
        """Create a new user"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            user_id = await conn.fetchval("""
                INSERT INTO users (email, password_hash, full_name, role)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, email, password_hash, full_name, role)
            return user_id
    
    async def update_user_password(self, user_id: int, password_hash: str):
        """Update user password"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE users 
                SET password_hash = $1, reset_token = NULL, reset_token_expires = NULL
                WHERE id = $2
            """, password_hash, user_id)
    
    async def set_reset_token(self, email: str, reset_token: str, expires_at: datetime):
        """Set password reset token"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE users 
                SET reset_token = $1, reset_token_expires = $2
                WHERE email = $3
            """, reset_token, expires_at, email)
    
    async def get_user_by_reset_token(self, reset_token: str) -> Optional[Dict]:
        """Get user by reset token"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, email, password_hash, full_name, role, is_active, reset_token_expires
                FROM users 
                WHERE reset_token = $1 AND reset_token_expires > NOW()
            """, reset_token)
            if row:
                return {
                    "id": row["id"],
                    "email": row["email"],
                    "password_hash": row["password_hash"],
                    "full_name": row["full_name"],
                    "role": row["role"],
                    "is_active": row["is_active"]
                }
            return None
    
    async def update_last_login(self, user_id: int):
        """Update user's last login time"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE users SET last_login = NOW() WHERE id = $1
            """, user_id)
    
    async def get_all_users(self) -> List[Dict]:
        """Get all users (for admin management)"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, email, full_name, role, is_active, created_at, last_login
                FROM users
                ORDER BY created_at DESC
            """)
            users = []
            for row in rows:
                users.append({
                    "id": row["id"],
                    "email": row["email"],
                    "full_name": row["full_name"],
                    "role": row["role"],
                    "is_active": row["is_active"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "last_login": row["last_login"].isoformat() if row["last_login"] else None
                })
            return users
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete a user (admin removal)"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM users WHERE id = $1
            """, user_id)
            return result == "DELETE 1"
    
    async def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics"""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Total violations
            total = await conn.fetchval("SELECT COUNT(*) FROM violations")
            
            # Today's violations
            today = await conn.fetchval("""
                SELECT COUNT(*) FROM violations 
                WHERE DATE(timestamp) = CURRENT_DATE
            """)
            
            # This week's violations
            week = await conn.fetchval("""
                SELECT COUNT(*) FROM violations 
                WHERE timestamp >= DATE_TRUNC('week', CURRENT_TIMESTAMP)
            """)
            
            # This month's violations
            month = await conn.fetchval("""
                SELECT COUNT(*) FROM violations 
                WHERE timestamp >= DATE_TRUNC('month', CURRENT_TIMESTAMP)
            """)
            
            # Top violation class
            top_class = await conn.fetchrow("""
                SELECT detection_class, COUNT(*) as count
                FROM violations
                GROUP BY detection_class
                ORDER BY count DESC
                LIMIT 1
            """)
            
            # Top 7 violations by date (most recent dates first)
            top_violations = await conn.fetch("""
                SELECT 
                    DATE(timestamp) as violation_date,
                    COUNT(*) as count
                FROM violations
                GROUP BY DATE(timestamp)
                ORDER BY violation_date DESC
                LIMIT 7
            """)
            
            # Top 7 locations/outlets by violation count
            top_outlets = await conn.fetch("""
                SELECT 
                    COALESCE(v.location, 'Unknown') as location,
                    COUNT(*) as count
                FROM violations v
                GROUP BY v.location
                ORDER BY count DESC
                LIMIT 7
            """)
            
            # Priority counts (High >= 0.85, Medium 0.70-0.85, Low < 0.70)
            # Prefer priority column, but fall back to confidence rules if priority is missing.
            high_priority = await conn.fetchval("""
                SELECT COUNT(*) FROM violations
                WHERE COALESCE(priority,
                    CASE
                        WHEN confidence >= 0.85 THEN 'high'
                        WHEN confidence >= 0.70 THEN 'medium'
                        ELSE 'low'
                    END
                ) = 'high'
            """)
            medium_priority = await conn.fetchval("""
                SELECT COUNT(*) FROM violations
                WHERE COALESCE(priority,
                    CASE
                        WHEN confidence >= 0.85 THEN 'high'
                        WHEN confidence >= 0.70 THEN 'medium'
                        ELSE 'low'
                    END
                ) = 'medium'
            """)
            low_priority = await conn.fetchval("""
                SELECT COUNT(*) FROM violations
                WHERE COALESCE(priority,
                    CASE
                        WHEN confidence >= 0.85 THEN 'high'
                        WHEN confidence >= 0.70 THEN 'medium'
                        ELSE 'low'
                    END
                ) = 'low'
            """)
            
            # Get unique detection classes for chart filters
            detection_classes = await conn.fetch("""
                SELECT DISTINCT detection_class
                FROM violations
                WHERE detection_class IS NOT NULL
                ORDER BY detection_class
            """)
            
            # Get violations summary for last 24 hours (hourly breakdown)
            hourly_summary = await conn.fetch("""
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) as count
                FROM violations
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour ASC
            """)
            
            return {
                "total_violations": total or 0,
                "today_violations": today or 0,
                "week_violations": week or 0,
                "month_violations": month or 0,
                "top_violation": {
                    "class": top_class["detection_class"] if top_class else "No Data",
                    "count": top_class["count"] if top_class else 0
                },
                "top_violations": [
                    {"date": row["violation_date"].isoformat() if hasattr(row["violation_date"], 'isoformat') else str(row["violation_date"]), "count": row["count"]}
                    for row in top_violations
                ],
                "detection_classes": [
                    row["detection_class"]
                    for row in detection_classes
                ],
                "top_outlets": [
                    {"location": row["location"], "count": row["count"]}
                    for row in top_outlets
                ],
                "high_priority": high_priority or 0,
                "medium_priority": medium_priority or 0,
                "low_priority": low_priority or 0,
                "hourly_summary": [
                    {
                        "hour": row["hour"].isoformat() if hasattr(row["hour"], 'isoformat') else str(row["hour"]),
                        "count": row["count"]
                    }
                    for row in hourly_summary
                ]
            }


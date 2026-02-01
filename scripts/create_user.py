"""
Script to create admin users in the database
Usage: python scripts/create_user.py <email> <password> <full_name>
Run from project root.
"""
import sys
import asyncio
from pathlib import Path
# Allow importing app when run as scripts/create_user.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.database import Database
from app.auth import get_password_hash

async def create_admin(email: str, password: str, full_name: str):
    """Create an admin user"""
    db = Database()
    
    # Initialize database (creates tables if they don't exist)
    await db.init_db()
    
    pool = await db.get_pool()
    
    # Check if user already exists
    existing_user = await db.get_user_by_email(email)
    if existing_user:
        print(f"❌ User with email {email} already exists!")
        return False
    
    # Hash password
    password_hash = get_password_hash(password)
    
    # Create user
    try:
        user_id = await db.create_user(email, password_hash, full_name, "admin")
        print(f"✅ Admin created successfully!")
        print(f"   ID: {user_id}")
        print(f"   Email: {email}")
        print(f"   Name: {full_name}")
        print(f"   Role: admin")
        return True
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/create_user.py <email> <password> <full_name>")
        print("\nExample:")
        print('  python create_user.py admin@company.com "SecurePass123" "Admin Name"')
        sys.exit(1)
    
    email = sys.argv[1]
    password = sys.argv[2]
    full_name = sys.argv[3]
    
    # Validate password length
    if len(password) < 6:
        print("❌ Password must be at least 6 characters long!")
        sys.exit(1)
    
    success = asyncio.run(create_admin(email, password, full_name))
    sys.exit(0 if success else 1)


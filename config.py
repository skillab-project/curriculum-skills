import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "skillcrawl"),
}

required_keys = ["host", "port", "user", "password", "database"]
if not all(DB_CONFIG[k] for k in required_keys):
    print("WARNING: Some DB configuration variables are missing. Check your .env or docker-compose settings.")

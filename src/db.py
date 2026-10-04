"""Database connection helpers. Settings come from the .env file in the project root."""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SETTINGS = {
    "user": os.getenv("POSTGRES_USER", "crash"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
    "dbname": os.getenv("POSTGRES_DB", "crash_risk"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}


def get_connection():
    """Raw psycopg2 connection, used for running SQL files and fast COPY loads."""
    return psycopg2.connect(**SETTINGS)


def get_engine():
    """SQLAlchemy engine, used with pandas: pd.read_sql("SELECT ...", get_engine())."""
    s = SETTINGS
    return create_engine(
        f"postgresql+psycopg2://{s['user']}:{s['password']}@{s['host']}:{s['port']}/{s['dbname']}"
    )


def run_sql_file(conn, path: Path):
    """Execute every statement in a .sql file."""
    with conn.cursor() as cur:
        cur.execute(path.read_text())
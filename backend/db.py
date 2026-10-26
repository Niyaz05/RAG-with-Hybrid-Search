"""
DAY 5 — PART A: Database models
=================================
Three tables, one clear relationship chain: users -> chat_sessions -> messages.

WHY SQLMODEL (not raw sqlite3, not a bigger ORM like SQLAlchemy directly)?
SQLModel lets one class define BOTH the database table AND the
FastAPI request/response shape. Less duplication than writing separate
Pydantic schemas and SQLAlchemy models for the same data, and it's what
FastAPI's own docs recommend for exactly this kind of project.

WHY SQLITE (not Postgres)?
Zero setup, file-based, plenty fast for a single/small user base with no
concurrent-write pressure. If this ever needed to scale, migrating is a
connection-string change (`sqlite:///...` -> `postgresql://...`), not a
schema rewrite - SQLModel's table definitions are the same either way.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
from sqlmodel import Field, Session, SQLModel, create_engine

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False is needed because FastAPI can handle requests
# on different threads, but SQLite connections are thread-bound by
# default - this is a standard, safe relaxation for SQLite specifically
# (would NOT be the right setting for a database like Postgres).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


# ---------------------------------------------------------------------------
# TABLE MODELS
# ---------------------------------------------------------------------------
# LEARNING NOTE: `table=True` tells SQLModel "this class IS a database
# table", not just a data-validation shape. Foreign keys (user_id,
# session_id) are how the three tables reference each other - this is
# the "relational" in relational database: rather than each message
# carrying a full copy of user info, it carries a small reference (an
# integer id) back to the one row that has it.

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    # Role gates access, not identity - see auth.py's require_admin().
    # New registrations always get "student" (see routes/auth_routes.py);
    # promoting to "admin" is a deliberate manual step, not something
    # anyone can request for themselves at signup.
    role: str = Field(default="student")
    created_at: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Kolkata")))


class ChatSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    title: str = Field(default="New chat")
    created_at: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Kolkata")))


class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id", index=True)
    role: str  # "user" or "assistant"
    content: str
    # Stored as a JSON string (list of [source, slide_label] pairs) rather
    # than a separate table - citations are small, read-only-after-write,
    # and always fetched alongside their message, so a normalized table
    # would add join complexity with no real benefit at this scale.
    citations_json: str = Field(default="[]")
    created_at: datetime = Field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Kolkata")))


# ---------------------------------------------------------------------------
# SETUP + SESSION DEPENDENCY
# ---------------------------------------------------------------------------

def init_db(engine_=engine):
    """Create all tables if they don't exist yet. Safe to call every startup."""
    SQLModel.metadata.create_all(engine_)


def get_session():
    """
    FastAPI dependency: yields a DB session for one request, closes it
    afterward automatically. Routes declare `session: Session =
    Depends(get_session)` to get a working DB handle without knowing how
    connections are managed.
    """
    with Session(engine) as session:
        yield session
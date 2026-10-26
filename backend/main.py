"""
DAY 5 — PART E: FastAPI app
==============================
Run with:  uvicorn main:app --reload   (from inside backend/)
Then visit http://127.0.0.1:8000/docs for the auto-generated, interactive
API docs - this is one of FastAPI's best features for learning: every
route you write shows up there automatically, with a working "try it out"
button, generated from your type hints and docstrings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # so `from db import ...` etc. work

from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI

from db import init_db
from routes.auth_routes import router as auth_router
from routes.chat_routes import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code before `yield` runs on startup; code after would run on shutdown.
    # This is the current recommended pattern, replacing the deprecated
    # @app.on_event("startup") decorator.
    init_db()
    yield


app = FastAPI(title="Ask My Course Docs API", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}
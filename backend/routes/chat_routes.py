
"""
DAY 5 — PART D: /chat and /sessions routes
=============================================
POST /chat            - protected. Sends a message, gets a cited answer,
                         persists both to the DB under the user's session.
GET  /sessions         - protected. List the current user's chat sessions.
GET  /sessions/{id}    - protected. Fetch a session's full message history
                         (only if it belongs to the requesting user).

DEPENDENCY INJECTION, ONE MORE TIME: the real answer-generation pipeline
(retrieve() + generate_answer() from Days 3-4) needs Ollama running. We
don't want THIS route's logic (auth check, saving messages, returning
JSON) to be untestable just because Ollama might not be available. So
`get_answer_fn` is itself a FastAPI dependency - the real one wires up
Days 3-4's pipeline, and tests override it with a fake via
`app.dependency_overrides[get_answer_fn] = lambda: fake_fn`.
"""

import json
from typing import Callable, Optional
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from sqlmodel import Session, select

from auth import get_current_user
from db import ChatSession, Message, User, get_session

router = APIRouter(tags=["chat"])

# An AnswerFn takes a query string and returns the same shape
# generate_answer() from Day 4 returns: {"answer", "citations_valid", ...}
AnswerFn = Callable[[str], dict]


def get_answer_fn() -> AnswerFn:
    """
    The REAL answer function, wiring together Day 2-4's pipeline. This is
    the ONLY place in this file that touches Ollama/Chroma/BM25 - imports
    are deliberately local (inside the function) so importing this module
    for tests doesn't require those pieces to even be installed correctly.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root / "generation"))
    sys.path.insert(0, str(root / "retrieval"))
    sys.path.insert(0, str(root / "ingest"))
    # pyrefly: ignore [missing-import]
    from prompt import generate_answer, ollama_chat
    # pyrefly: ignore [missing-import]
    from search import load_bm25_index, load_vector_index
    # pyrefly: ignore [missing-import]
    from hybrid_retrieve import retrieve, build_chunk_lookup

    bm25, chunks = load_bm25_index()
    collection = load_vector_index()
    lookup = build_chunk_lookup(chunks)

    def answer(query: str) -> dict:
        retrieved = retrieve(query, bm25, chunks, collection, lookup)
        return generate_answer(query, retrieved, chat_fn=ollama_chat)

    return answer


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None  # omit to start a new session


@router.post("/chat")
def chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    answer_fn: AnswerFn = Depends(get_answer_fn),
):
    # Resolve or create the chat session, and confirm ownership if one was given.
    if body.session_id is not None:
        chat_session = session.get(ChatSession, body.session_id)
        if chat_session is None or chat_session.user_id != user.id:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        chat_session = ChatSession(user_id=user.id, title=body.message[:50])
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)

    # Save the user's message.
    session.add(Message(session_id=chat_session.id, role="user", content=body.message))

    # Get the cited answer (this is where Ollama gets called, via the injected answer_fn).
    result = answer_fn(body.message)

    # Save the assistant's message, with citations serialized to JSON.
    session.add(
        Message(
            session_id=chat_session.id,
            role="assistant",
            content=result["answer"],
            citations_json=json.dumps(result.get("citations_valid", [])),
        )
    )
    session.commit()

    return {
        "session_id": chat_session.id,
        "answer": result["answer"],
        "citations_valid": result.get("citations_valid", []),
        "citations_invalid": result.get("citations_invalid", []),
    }


@router.get("/sessions")
def list_sessions(user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    sessions = session.exec(
        select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.created_at.desc())
    ).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]


@router.get("/sessions/{session_id}")
def get_session_messages(
    session_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    chat_session = session.get(ChatSession, session_id)
    if chat_session is None or chat_session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = session.exec(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    ).all()
    return [
        {
            "role": m.role,
            "content": m.content,
            "citations": json.loads(m.citations_json),
            "created_at": m.created_at,
        }
        for m in messages
    ]
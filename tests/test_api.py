"""
TESTS for the full backend API
=================================
Run with:  pytest tests/test_api.py -v

TWO key testing tricks used here, both extending patterns from earlier days:

1. ISOLATED TEST DATABASE: real API tests must not touch your real
   data/app.db (that would pollute your actual user data with test
   accounts every time you run pytest). We create a fresh in-memory
   SQLite database per test and override the get_session dependency to
   use it - FastAPI's dependency_overrides makes this a couple of lines.

2. FAKE answer_fn: chat_routes.py's real get_answer_fn() needs Ollama.
   We override THAT dependency too with an instant fake, exactly like
   embed_fn/chat_fn on earlier days - proving the auth + persistence
   logic works without needing Ollama running during tests.
"""

import sys
from pathlib import Path

import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
# pyrefly: ignore [missing-import]
from sqlmodel import Session, SQLModel, create_engine
# pyrefly: ignore [missing-import]
from sqlmodel.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# pyrefly: ignore [missing-import]
from db import get_session
# pyrefly: ignore [missing-import]
from main import app
# pyrefly: ignore [missing-import]
from routes.chat_routes import get_answer_fn


@pytest.fixture(name="client")
def client_fixture():
    """
    Builds a fresh in-memory DB + TestClient for EACH test function, so
    tests never see leftover data from a previous test. StaticPool keeps
    the same in-memory DB connection alive for the whole test (in-memory
    SQLite DBs normally vanish the instant a connection closes).
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)

    def get_session_override():
        with Session(engine) as session:
            yield session

    def fake_answer_fn():
        def answer(query: str) -> dict:
            return {
                "answer": f"Fake answer about: {query} [april_2026.pdf, slide 7]",
                "citations_valid": [("april_2026.pdf", "slide 7")],
                "citations_invalid": [],
            }
        return answer

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_answer_fn] = fake_answer_fn

    yield TestClient(app)

    app.dependency_overrides.clear()


def _register_and_login(client, email="student@example.com", password="password123"):
    client.post("/auth/register", params={"email": email, "password": password})
    resp = client.post("/auth/login", data={"username": email, "password": password})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# --- registration / login -----------------------------------------------------

def test_register_creates_a_student_by_default(client):
    resp = client.post("/auth/register", params={"email": "a@example.com", "password": "pw12345"})
    assert resp.status_code == 201
    assert resp.json()["role"] == "student"


def test_cannot_register_the_same_email_twice(client):
    client.post("/auth/register", params={"email": "dupe@example.com", "password": "pw12345"})
    resp = client.post("/auth/register", params={"email": "dupe@example.com", "password": "pw12345"})
    assert resp.status_code == 400


def test_login_with_correct_credentials_returns_a_token(client):
    client.post("/auth/register", params={"email": "b@example.com", "password": "pw12345"})
    resp = client.post("/auth/login", data={"username": "b@example.com", "password": "pw12345"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_with_wrong_password_is_rejected(client):
    client.post("/auth/register", params={"email": "c@example.com", "password": "pw12345"})
    resp = client.post("/auth/login", data={"username": "c@example.com", "password": "wrongpw"})
    assert resp.status_code == 401


# --- protected routes require auth --------------------------------------------

def test_chat_without_token_is_rejected(client):
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 401


def test_sessions_without_token_is_rejected(client):
    resp = client.get("/sessions")
    assert resp.status_code == 401


# --- chat + persistence ---------------------------------------------------------

def test_chat_creates_a_new_session_and_returns_the_fake_answer(client):
    headers = _register_and_login(client)
    resp = client.post("/chat", json={"message": "Why fx and fy?"}, headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert "Fake answer about: Why fx and fy?" in body["answer"]
    assert body["citations_valid"] == [["april_2026.pdf", "slide 7"]]
    assert body["session_id"] is not None


def test_second_message_with_same_session_id_reuses_the_session(client):
    headers = _register_and_login(client)
    first = client.post("/chat", json={"message": "first question"}, headers=headers).json()
    second = client.post(
        "/chat", json={"message": "follow up", "session_id": first["session_id"]}, headers=headers
    ).json()

    assert second["session_id"] == first["session_id"]


def test_sessions_list_shows_the_users_own_sessions(client):
    headers = _register_and_login(client)
    client.post("/chat", json={"message": "q1"}, headers=headers)
    client.post("/chat", json={"message": "q2"}, headers=headers)

    resp = client.get("/sessions", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_session_messages_include_user_and_assistant_turns_in_order(client):
    headers = _register_and_login(client)
    chat_resp = client.post("/chat", json={"message": "Why fx and fy?"}, headers=headers).json()

    resp = client.get(f"/sessions/{chat_resp['session_id']}", headers=headers)
    messages = resp.json()

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Why fx and fy?"
    assert messages[1]["citations"] == [["april_2026.pdf", "slide 7"]]


def test_user_cannot_access_another_users_session(client):
    """
    Security check: user A's session_id must be invisible to user B, even
    though both are validly logged in. 404, not 403 - we don't want to
    even confirm the session exists to someone who doesn't own it.
    """
    headers_a = _register_and_login(client, email="alice@example.com")
    headers_b = _register_and_login(client, email="bob@example.com")

    session_id = client.post("/chat", json={"message": "alice's question"}, headers=headers_a).json()["session_id"]

    resp = client.get(f"/sessions/{session_id}", headers=headers_b)
    assert resp.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
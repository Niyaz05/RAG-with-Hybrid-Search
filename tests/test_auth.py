"""
TESTS for backend/auth.py
============================
Run with:  pytest tests/test_auth.py -v
No FastAPI app or DB needed here - just the hashing and token functions
in isolation.
"""

import sys
import time
from pathlib import Path

import pytest
from jose import JWTError

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# pyrefly: ignore [missing-import]
from auth import hash_password, verify_password, create_access_token, decode_access_token


# --- password hashing --------------------------------------------------------

def test_hash_password_does_not_return_plaintext():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"


def test_verify_password_accepts_correct_password():
    hashed = hash_password("mypassword123")
    assert verify_password("mypassword123", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("mypassword123")
    assert verify_password("wrongpassword", hashed) is False


def test_same_password_hashes_differently_each_time():
    """
    LEARNING NOTE: bcrypt includes a random salt in every hash it
    generates, so hashing the SAME password twice produces DIFFERENT
    output strings. This is intentional - it defeats precomputed
    "rainbow table" attacks. Both hashes still verify correctly against
    the original password, though.
    """
    h1 = hash_password("samepassword")
    h2 = hash_password("samepassword")
    assert h1 != h2
    assert verify_password("samepassword", h1)
    assert verify_password("samepassword", h2)


# --- JWT ----------------------------------------------------------------------

def test_create_and_decode_token_roundtrip():
    token = create_access_token(data={"sub": "student@example.com", "role": "student"})
    payload = decode_access_token(token)

    assert payload["sub"] == "student@example.com"
    assert payload["role"] == "student"


def test_decode_rejects_tampered_token():
    token = create_access_token(data={"sub": "student@example.com"})
    tampered = token[:-4] + "abcd"  # corrupt the signature

    with pytest.raises(JWTError):
        decode_access_token(tampered)


def test_decode_rejects_garbage_string():
    with pytest.raises(JWTError):
        decode_access_token("this.is.not.a.real.token")


def test_expired_token_is_rejected():
    from datetime import timedelta
    # Issue a token that expired 1 second ago.
    token = create_access_token(data={"sub": "x@example.com"}, expires_delta=timedelta(seconds=-1))

    with pytest.raises(JWTError):
        decode_access_token(token)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
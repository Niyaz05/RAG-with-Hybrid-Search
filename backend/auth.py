"""
DAY 5 — PART B: Authentication
=================================
Three responsibilities:
  1. Hash/verify passwords (bcrypt via passlib) - never store or compare
     plaintext passwords.
  2. Issue/decode JWTs (python-jose) - stateless proof of identity.
  3. FastAPI dependencies that routes use to require "must be logged in"
     or "must be an admin".

WHY BCRYPT SPECIFICALLY: it's a ONE-WAY hash with a built-in random salt
per password, and it's deliberately slow (tunable "work factor") to
resist brute-force guessing. You can verify a password against its hash;
you can never reverse a hash back into the original password - that's
the whole point.

WHY JWT IS "STATELESS": the token itself contains everything needed to
verify it (the user's identity + a cryptographic signature using
SECRET_KEY). The server doesn't need to look up "is this session still
valid?" in a database on every request - it just re-computes the
signature and checks it matches. Downside: you can't easily revoke a
single token early without extra machinery (a blocklist) - a simplification
worth naming if asked about it in an interview.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

# pyrefly: ignore [missing-import]
from fastapi import Depends, HTTPException, status
# pyrefly: ignore [missing-import]
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt #jose stands for JavaScript Object Signing and Encryption
# pyrefly: ignore [missing-import]
from passlib.context import CryptContext
# pyrefly: ignore [missing-import]
from sqlmodel import Session, select

from db import User, get_session

# LEARNING NOTE: in a real deployment this MUST come from an environment
# variable / secrets manager, never hardcoded - anyone with this string
# can forge valid tokens for any user. The os.environ.get fallback here
# is only acceptable because this is a local-only personal study tool.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours - generous for a personal tool

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# tokenUrl tells FastAPI's auto-generated docs (/docs) where to send a
# login request from the "Authorize" button - it doesn't affect the
# actual login route's behavior.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


# ---------------------------------------------------------------------------
# PASSWORD HASHING — pure-ish, easy to test directly
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# JWT ISSUING / DECODING
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    data typically looks like {"sub": user.email, "role": user.role} -
    "sub" (subject) is the JWT-standard field name for "who this token is about".
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises JWTError if the token is invalid, tampered with, or expired."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ---------------------------------------------------------------------------
# FASTAPI DEPENDENCIES — what routes actually declare with Depends(...)
# ---------------------------------------------------------------------------

def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    """
    Validates the bearer token from the Authorization header and returns
    the corresponding User row. Any route with
    `user: User = Depends(get_current_user)` is now "login required" -
    FastAPI runs this BEFORE the route's own code, and rejects the
    request with 401 automatically if it raises.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
        if email is None:
            raise unauthorized
    except JWTError:
        raise unauthorized

    user = session.exec(select(User).where(User.email == email)).first()
    if user is None:
        raise unauthorized
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    Stacks on top of get_current_user: must be logged in AND have the
    admin role. This is the dependency Day 6's document-upload endpoint
    will use - uploading/re-indexing is expensive and destructive, so it
    should not be reachable by any logged-in user.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for this action",
        )
    return user
"""
Authentification et gestion des rôles (admin / chauffeur).

JWT sans état (stateless) : le token contient l'id, l'email et le rôle de
l'utilisateur, signé avec JWT_SECRET (voir .env). Chaque route protégée
vérifie le token via la dependency `require_role(...)`.
"""
import logging
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, Header, HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logger = logging.getLogger("velib.auth")

DB_URL = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
)
engine = create_engine(DB_URL)

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "480"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ROLES = {"admin", "chauffeur"}


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateUserRequest(BaseModel):
    email: str
    password: str
    full_name: str | None = None
    role: str = "chauffeur"


class UpdateUserRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    full_name: str | None = None


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    role: str
    is_active: bool


def ensure_auth_tables():
    """Table des comptes (admin / chauffeur) + admin par défaut si besoin."""
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS auth"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS auth.users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT NOT NULL DEFAULT 'chauffeur',
                is_active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        admin_exists = conn.execute(
            text("select 1 from auth.users where role = 'admin' limit 1")
        ).first()
        if not admin_exists:
            default_email = os.environ.get("ADMIN_EMAIL", "admin@velib.local")
            default_password = os.environ.get("ADMIN_PASSWORD", "admin123")
            conn.execute(text("""
                INSERT INTO auth.users (email, password_hash, full_name, role)
                VALUES (:email, :password_hash, 'Administrateur', 'admin')
                ON CONFLICT (email) DO NOTHING
            """), {
                "email": default_email,
                "password_hash": pwd_context.hash(default_password),
            })
            logger.warning(
                "Compte admin par défaut créé : %s (mot de passe : voir ADMIN_PASSWORD dans .env) "
                "— change-le après la première connexion.",
                default_email,
            )


def authenticate_user(email: str, password: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("select * from auth.users where email = :email and is_active = true"),
            {"email": email},
        ).mappings().first()
    if not row or not pwd_context.verify(password, row["password_hash"]):
        # On ne journalise jamais le mot de passe, seulement l'email tenté :
        # utile pour repérer un brute-force sans exposer de secret dans les logs.
        logger.warning("Échec de connexion pour %s", email)
        return None
    logger.info("Connexion réussie : %s (rôle %s)", row["email"], row["role"])
    return dict(row)


def create_access_token(user: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "role": user["role"],
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: str = Header(default=None)) -> dict:
    """Décode le JWT envoyé dans le header `Authorization: Bearer <token>`."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expirée, reconnecte-toi")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
    return {"id": int(payload["sub"]), "email": payload["email"], "role": payload["role"]}


def require_role(*allowed_roles: str):
    """Dependency factory : protège une route pour un ou plusieurs rôles."""
    def dependency(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès réservé au(x) rôle(s) : {', '.join(allowed_roles)}",
            )
        return user
    return dependency

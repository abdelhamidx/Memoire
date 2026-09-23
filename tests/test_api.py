"""
Tests de l'API Vélib' Prévision.

Ces tests tapent la vraie base de données (pas de mock) : ils vérifient
que le pipeline complet (collecte -> bronze -> dbt -> silver -> API)
fonctionne bout en bout. Ils supposent que la base PostgreSQL est
accessible (docker compose up -d db) et que dbt a été exécuté au moins
une fois (ou que tests/ci_seed.sql a été chargé, voir CI).

Lancer : pytest tests/
"""

import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# Le compte admin par défaut est créé automatiquement au démarrage de l'API
# (voir api/auth.py::ensure_auth_tables) avec ces identifiants, sauf si
# ADMIN_EMAIL / ADMIN_PASSWORD sont surchargés dans l'environnement.
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@velib.local")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


@pytest.fixture(scope="module")
def admin_token():
    """Connecte le compte admin par défaut une fois, réutilisé par les tests admin."""
    response = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert response.status_code == 200, "Le compte admin par défaut doit exister et fonctionner"
    return response.json()["access_token"]


@pytest.fixture(scope="module")
def chauffeur_token(admin_token):
    """Crée (si besoin) et connecte un compte chauffeur de test, pour les tests de rôle."""
    email = "chauffeur.test@velib.local"
    password = "chauffeur123"
    headers = {"Authorization": f"Bearer {admin_token}"}

    client.post(
        "/admin/users",
        json={"email": email, "password": password, "full_name": "Chauffeur Test", "role": "chauffeur"},
        headers=headers,
    )  # 201 la première fois, 409 (déjà créé) les fois suivantes : les deux sont acceptables ici

    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


# --- Santé / pages -----------------------------------------------------

def test_health():
    """L'API doit répondre que tout va bien."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_homepage_serves_map():
    """La racine doit servir la page HTML de la carte."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# --- Stations ------------------------------------------------------------

def test_list_stations():
    """La liste des stations doit renvoyer un tableau non vide avec les bons champs."""
    response = client.get("/stations")
    assert response.status_code == 200
    stations = response.json()
    assert isinstance(stations, list)
    assert len(stations) > 0

    station = stations[0]
    for field in ["station_id", "name", "lat", "lon", "capacity"]:
        assert field in station


def test_get_single_station():
    """Le détail d'une station existante doit fonctionner (round-trip via /stations)."""
    stations = client.get("/stations").json()
    assert len(stations) > 0
    station_id = stations[0]["station_id"]

    response = client.get(f"/stations/{station_id}")
    assert response.status_code == 200
    assert response.json()["station_id"] == station_id


def test_get_unknown_station_returns_404():
    """Une station inexistante doit renvoyer une 404, pas une erreur 500."""
    response = client.get("/stations/ID_QUI_NEXISTE_PAS")
    assert response.status_code == 404


def test_station_history():
    """L'historique d'une station doit être une liste triée chronologiquement."""
    stations = client.get("/stations").json()
    station_id = stations[0]["station_id"]

    response = client.get(f"/stations/{station_id}/history?limit=10")
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)

    if len(history) > 1:
        timestamps = [h["collected_at"] for h in history]
        assert timestamps == sorted(timestamps)  # ordre chronologique croissant


# --- Prédictions (legacy + multi-horizon) --------------------------------

def test_predictions_batch():
    """L'endpoint de prédiction en lot doit renvoyer une prédiction par station éligible."""
    response = client.get("/predictions")
    # 503 tant que le modèle n'est pas encore entraîné/disponible : acceptable.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        predictions = response.json()
        assert isinstance(predictions, list)
        if predictions:
            assert "predicted_num_bikes_available" in predictions[0]


@pytest.mark.parametrize("horizon", [15, 30, 60])
def test_predictions_batch_valid_horizon(horizon):
    """Un horizon valide (15/30/60) ne doit jamais renvoyer 422, même si le modèle
    horizon n'existe pas encore (repli automatique sur la persistance)."""
    response = client.get(f"/predictions?horizon_minutes={horizon}")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        predictions = response.json()
        assert isinstance(predictions, list)


def test_predictions_batch_invalid_horizon_returns_422():
    """Un horizon hors de {15, 30, 60} doit être rejeté proprement (422), pas planter."""
    response = client.get("/predictions?horizon_minutes=45")
    assert response.status_code == 422


def test_predict_single_station_default_horizon():
    """Prédiction unitaire sans horizon : comportement legacy (~5 min)."""
    stations = client.get("/stations").json()
    station_id = stations[0]["station_id"]

    response = client.get(f"/predict/{station_id}")
    assert response.status_code in (200, 422, 503)
    if response.status_code == 200:
        body = response.json()
        assert "predicted_num_bikes_available" in body
        assert body["horizon_minutes"] == 5


@pytest.mark.parametrize("horizon", [15, 30, 60])
def test_predict_single_station_multi_horizon(horizon):
    """Prédiction unitaire à un horizon donné : doit répondre et refléter l'horizon demandé,
    en se rabattant sur la persistance si aucun modèle ne bat la baseline pour cet horizon
    (voir ml_models/metrics.json)."""
    stations = client.get("/stations").json()
    station_id = stations[0]["station_id"]

    response = client.get(f"/predict/{station_id}?horizon_minutes={horizon}")
    assert response.status_code in (200, 422, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["horizon_minutes"] == horizon
        assert "predicted_num_bikes_available" in body
        assert "model" in body


def test_predict_single_station_invalid_horizon_returns_422():
    stations = client.get("/stations").json()
    station_id = stations[0]["station_id"]

    response = client.get(f"/predict/{station_id}?horizon_minutes=999")
    assert response.status_code == 422


# --- Rééquilibrage ---------------------------------------------------------

def test_rebalancing_requires_no_auth_but_works():
    """L'endpoint de suggestions de rééquilibrage doit répondre (liste, potentiellement vide)."""
    response = client.get("/rebalancing")
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert isinstance(response.json(), list)


def test_rebalancing_invalid_horizon_returns_422():
    response = client.get("/rebalancing?horizon_minutes=7")
    assert response.status_code == 422


# --- Authentification et rôles --------------------------------------------

def test_login_wrong_password_returns_401():
    response = client.post("/auth/login", json={"email": ADMIN_EMAIL, "password": "mot_de_passe_incorrect"})
    assert response.status_code == 401


def test_login_unknown_email_returns_401():
    response = client.post("/auth/login", json={"email": "personne@nulle-part.fake", "password": "x"})
    assert response.status_code == 401


def test_login_success_returns_token(admin_token):
    assert isinstance(admin_token, str)
    assert len(admin_token) > 20


def test_auth_me_without_token_returns_401():
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_auth_me_with_token(admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


def test_auth_me_with_invalid_token_returns_401():
    headers = {"Authorization": "Bearer token.invalide.evidemment"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401


def test_admin_users_requires_auth():
    """Sans token, la liste des utilisateurs ne doit jamais être accessible."""
    response = client.get("/admin/users")
    assert response.status_code == 401


def test_admin_users_forbidden_for_chauffeur(chauffeur_token):
    """Un compte chauffeur ne doit pas pouvoir lister les utilisateurs (réservé admin)."""
    headers = {"Authorization": f"Bearer {chauffeur_token}"}
    response = client.get("/admin/users", headers=headers)
    assert response.status_code == 403


def test_admin_users_allowed_for_admin(admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.get("/admin/users", headers=headers)
    assert response.status_code == 200
    users = response.json()
    assert isinstance(users, list)
    assert any(u["role"] == "admin" for u in users)


def test_create_user_duplicate_email_returns_409(admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "email": "chauffeur.test@velib.local",  # créé par la fixture chauffeur_token
        "password": "peu-importe",
        "full_name": "Doublon",
        "role": "chauffeur",
    }
    # S'assure que le compte existe déjà (via la fixture), puis vérifie le conflit.
    client.post("/admin/users", json=payload, headers=headers)
    response = client.post("/admin/users", json=payload, headers=headers)
    assert response.status_code == 409


def test_create_user_invalid_role_returns_422(admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    response = client.post(
        "/admin/users",
        json={"email": "role.invalide@velib.local", "password": "x", "role": "super-admin"},
        headers=headers,
    )
    assert response.status_code == 422

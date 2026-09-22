"""
Tests de l'API Vélib' Prévision.

Ces tests tapent la vraie base de données (pas de mock) : ils vérifient
que le pipeline complet (collecte -> bronze -> dbt -> silver -> API)
fonctionne bout en bout. Ils supposent que la base PostgreSQL est
accessible (docker compose up -d db) et que dbt a été exécuté au moins
une fois.

Lancer : pytest tests/
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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

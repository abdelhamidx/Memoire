"""
API FastAPI qui sert l'état des stations Vélib' (et plus tard les prédictions).

Lancer en local : uvicorn api.main:app --reload
Doc interactive : http://localhost:8000/docs
"""

import os
from datetime import datetime, timezone

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
)
engine = create_engine(DB_URL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "xgb_baseline.joblib")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

N_LAGS = 3

app = FastAPI(title="Vélib' Prévision API")

# Sert la carte interactive (frontend/index.html) et ses futurs assets
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_map():
    """Page d'accueil : la carte interactive des stations."""
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations")
def list_stations():
    """Liste toutes les stations avec leur dernier état connu (couche silver)."""
    query = text("""
        select
            i.station_id,
            i.name,
            i.lat,
            i.lon,
            i.capacity,
            s.num_bikes_available,
            s.num_docks_available,
            s.collected_at
        from dbt_dev.silver_station_information i
        left join lateral (
            select num_bikes_available, num_docks_available, collected_at
            from dbt_dev.silver_station_status ss
            where ss.station_id = i.station_id
            order by ss.collected_at desc
            limit 1
        ) s on true
        order by i.station_id
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(row) for row in rows]


@app.get("/stations/{station_id}")
def get_station(station_id: str):
    """Détail d'une station : infos statiques + dernier état."""
    query = text("""
        select
            i.station_id,
            i.name,
            i.lat,
            i.lon,
            i.capacity,
            s.num_bikes_available,
            s.num_docks_available,
            s.collected_at
        from dbt_dev.silver_station_information i
        left join lateral (
            select num_bikes_available, num_docks_available, collected_at
            from dbt_dev.silver_station_status ss
            where ss.station_id = i.station_id
            order by ss.collected_at desc
            limit 1
        ) s on true
        where i.station_id = :station_id
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"station_id": station_id}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Station introuvable")
    return dict(row)


@app.get("/predict/{station_id}")
def predict_station(station_id: str):
    """
    Prédit le nombre de vélos disponibles à la prochaine collecte pour une
    station, à partir du modèle XGBoost entraîné (notebooks/02_...).

    Le modèle utilise : heure, jour de la semaine, et les 3 dernières
    valeurs connues (lags). Il est réentraîné manuellement au fur et à
    mesure que l'historique de collecte s'enrichit (voir le notebook).
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible (ml_models/xgb_baseline.joblib introuvable). "
                   "Entraîne-le d'abord via notebooks/02_features_and_baseline_model.ipynb.",
        )

    query = text("""
        select collected_at, num_bikes_available
        from dbt_dev.silver_station_status
        where station_id = :station_id
        order by collected_at desc
        limit :n_lags
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"station_id": station_id, "n_lags": N_LAGS}).mappings().all()

    if len(rows) < N_LAGS:
        raise HTTPException(
            status_code=422,
            detail=f"Pas assez d'historique pour cette station ({len(rows)}/{N_LAGS} collectes disponibles).",
        )

    now = rows[0]["collected_at"]
    lags = [row["num_bikes_available"] for row in rows]  # [lag_1, lag_2, lag_3]

    features = [[now.hour, now.weekday(), lags[0], lags[1], lags[2]]]
    prediction = model.predict(features)[0]

    return {
        "station_id": station_id,
        "based_on_collected_at": now,
        "current_num_bikes_available": lags[0],
        "predicted_num_bikes_available": round(float(prediction), 1),
        "model": "xgboost_baseline",
    }

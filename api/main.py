"""
API FastAPI qui sert l'état des stations Vélib' (et plus tard les prédictions).

Lancer en local : uvicorn api.main:app --reload
Doc interactive : http://localhost:8000/docs
"""

import os

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


# TODO: endpoint /predict/{station_id} qui renvoie la disponibilité prédite
# (une fois le modèle entraîné — étape 11-12 du cahier des charges)

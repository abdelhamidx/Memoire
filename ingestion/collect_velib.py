"""
Collecte des données Vélib' Métropole (API GBFS) vers la couche bronze PostgreSQL.

Usage:
    python ingestion/collect_velib.py            # une seule collecte
    python ingestion/collect_velib.py --loop      # collecte en continu (toutes les COLLECT_INTERVAL_MINUTES)

Étape 6-7 du cahier des charges : ce script doit tourner en continu le plus
tôt possible dans le projet, car il faut plusieurs semaines d'historique
avant de pouvoir entraîner un modèle de prévision.
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STATION_STATUS_URL = os.environ["VELIB_STATION_STATUS_URL"]
STATION_INFO_URL = os.environ["VELIB_STATION_INFO_URL"]
COLLECT_INTERVAL_MINUTES = int(os.environ.get("COLLECT_INTERVAL_MINUTES", 5))

DB_URL = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
)


def get_engine():
    return create_engine(DB_URL)


def ensure_bronze_tables(engine):
    """Crée les tables bronze si elles n'existent pas encore."""
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bronze.station_status_raw (
                id SERIAL PRIMARY KEY,
                collected_at TIMESTAMPTZ NOT NULL,
                station_id TEXT NOT NULL,
                num_bikes_available INT,
                num_docks_available INT,
                is_installed INT,
                is_renting INT,
                is_returning INT,
                last_reported BIGINT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bronze.station_information_raw (
                station_id TEXT PRIMARY KEY,
                name TEXT,
                lat DOUBLE PRECISION,
                lon DOUBLE PRECISION,
                capacity INT,
                collected_at TIMESTAMPTZ NOT NULL
            )
        """))


def fetch_station_status() -> list[dict]:
    resp = requests.get(STATION_STATUS_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()["data"]["stations"]


def fetch_station_information() -> list[dict]:
    resp = requests.get(STATION_INFO_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()["data"]["stations"]


def store_station_status(engine, stations: list[dict], collected_at: datetime):
    rows = [
        {
            "collected_at": collected_at,
            "station_id": s["station_id"],
            "num_bikes_available": s.get("num_bikes_available"),
            "num_docks_available": s.get("num_docks_available"),
            "is_installed": s.get("is_installed"),
            "is_renting": s.get("is_renting"),
            "is_returning": s.get("is_returning"),
            "last_reported": s.get("last_reported"),
        }
        for s in stations
    ]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bronze.station_status_raw
                (collected_at, station_id, num_bikes_available, num_docks_available,
                 is_installed, is_renting, is_returning, last_reported)
            VALUES
                (:collected_at, :station_id, :num_bikes_available, :num_docks_available,
                 :is_installed, :is_renting, :is_returning, :last_reported)
        """), rows)
    logger.info("✓ %d lignes station_status insérées", len(rows))


def store_station_information(engine, stations: list[dict], collected_at: datetime):
    rows = [
        {
            "station_id": s["station_id"],
            "name": s.get("name"),
            "lat": s.get("lat"),
            "lon": s.get("lon"),
            "capacity": s.get("capacity"),
            "collected_at": collected_at,
        }
        for s in stations
    ]
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bronze.station_information_raw (station_id, name, lat, lon, capacity, collected_at)
            VALUES (:station_id, :name, :lat, :lon, :capacity, :collected_at)
            ON CONFLICT (station_id) DO UPDATE SET
                name = EXCLUDED.name, lat = EXCLUDED.lat, lon = EXCLUDED.lon,
                capacity = EXCLUDED.capacity, collected_at = EXCLUDED.collected_at
        """), rows)
    logger.info("✓ %d stations (information) mises à jour", len(rows))


def collect_once(engine):
    now = datetime.now(timezone.utc)
    logger.info("Collecte à %s", now.isoformat())
    store_station_status(engine, fetch_station_status(), now)
    store_station_information(engine, fetch_station_information(), now)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Collecte en continu")
    args = parser.parse_args()

    engine = get_engine()
    ensure_bronze_tables(engine)

    if not args.loop:
        collect_once(engine)
        return

    logger.info("Démarrage de la collecte en continu (toutes les %d min). Ctrl+C pour arrêter.",
                COLLECT_INTERVAL_MINUTES)
    while True:
        try:
            collect_once(engine)
        except Exception:
            logger.exception("Erreur pendant la collecte, on continue à la prochaine itération")
        time.sleep(COLLECT_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()

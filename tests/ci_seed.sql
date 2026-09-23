-- Jeu de données minimal pour que les tests pytest (qui tapent la vraie
-- base) aient de quoi répondre en CI, sans avoir à faire tourner tout le
-- pipeline dbt (qui a besoin de vraies données collectées).
CREATE SCHEMA IF NOT EXISTS dbt_dev;

CREATE TABLE dbt_dev.silver_station_information (
    station_id TEXT PRIMARY KEY,
    name TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    capacity INT
);

CREATE TABLE dbt_dev.silver_station_status (
    station_id TEXT,
    collected_at TIMESTAMPTZ,
    num_bikes_available INT,
    num_docks_available INT
);

INSERT INTO dbt_dev.silver_station_information (station_id, name, lat, lon, capacity) VALUES
    ('ci_station_1', 'Station Test CI 1', 48.8566, 2.3522, 20),
    ('ci_station_2', 'Station Test CI 2', 48.8606, 2.3376, 15);

INSERT INTO dbt_dev.silver_station_status (station_id, collected_at, num_bikes_available, num_docks_available) VALUES
    ('ci_station_1', now() - interval '2 hour', 5, 15),
    ('ci_station_1', now() - interval '1 hour', 7, 13),
    ('ci_station_1', now(), 8, 12),
    ('ci_station_2', now() - interval '2 hour', 3, 12),
    ('ci_station_2', now() - interval '1 hour', 4, 11),
    ('ci_station_2', now(), 6, 9);

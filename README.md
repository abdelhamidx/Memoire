# PFE — Prévision & optimisation de la disponibilité Vélib'

Projet de fin d'études (Master Data/IA/Dev) : prévision spatio-temporelle de la disponibilité des stations Vélib' Métropole et détection des stations à risque de pénurie/saturation.

📄 Cahier des charges complet : voir le document Claude partagé (architecture, sources de données, planning, checklist de démarrage).

## Structure du projet

```
.
├── ingestion/       # Collecte des données (API Vélib' + météo) → bronze
├── dbt_velib/       # Transformations bronze → silver → gold (dbt)
├── api/             # API FastAPI qui sert les prédictions
├── frontend/        # Interface carte interactive
├── docs/            # Notes, schémas, exports
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Démarrage rapide

1. Copier `.env.example` vers `.env` et renseigner les valeurs
2. `python -m venv venv && source venv/bin/activate` (ou `venv\Scripts\activate` sous Windows)
3. `pip install -r requirements.txt`
4. `docker compose up -d db` (démarre uniquement PostgreSQL pour commencer)
5. Lancer la collecte : `python ingestion/collect_velib.py`

## Suivi d'avancement

Voir la checklist "Guide de démarrage — étapes concrètes" dans le cahier des charges.

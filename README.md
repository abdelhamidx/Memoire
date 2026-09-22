# Vélib' Prévision — PFE

Système de prévision de la disponibilité des stations Vélib' Métropole (Paris) : collecte continue des données temps réel, pipeline de transformation, modèle de machine learning, API et carte interactive.

Projet de fin d'études (Master Data / IA / Dev).

## Aperçu

- Collecte automatique des données officielles Vélib' (GBFS / Smovengo) toutes les 5 minutes, en continu
- Pipeline de données en couches (bronze → silver) avec **dbt**
- Modèle de prévision (**XGBoost**) : prédit le nombre de vélos disponibles à la prochaine collecte
- **API REST** (FastAPI) exposant l'état des stations et les prédictions
- **Carte interactive** temps réel : état actuel, prédictions, historique par station, alertes stations vides
- Tout le pipeline est **conteneurisé** (Docker Compose) : `docker compose up` suffit à faire tourner l'ensemble

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────┐     ┌───────────┐
│  API Vélib'      │────▶│  ingestion   │────▶│ PostgreSQL   │────▶│   dbt   │────▶│    API    │
│  (Smovengo GBFS) │     │  (Docker,    │     │  bronze      │     │ bronze  │     │  FastAPI  │
└─────────────────┘     │  toutes les  │     │  (raw)       │     │  →      │     │  + modèle │
                         │   5 min)     │     └─────────────┘     │ silver  │     │  XGBoost  │
                         └──────────────┘                          └─────────┘     └─────┬─────┘
                                                                                           │
                                                                                           ▼
                                                                                  ┌──────────────────┐
                                                                                  │ Carte interactive │
                                                                                  │ (Leaflet, temps   │
                                                                                  │ réel + prédiction)│
                                                                                  └──────────────────┘
```

## Structure du projet

```
.
├── ingestion/            # Script de collecte (API Vélib') → table bronze, dockerisé (restart: always)
│   ├── collect_velib.py
│   └── Dockerfile
├── dbt_velib/dbt_velib/  # Projet dbt : transformations bronze → silver
│   └── models/
│       ├── bronze/       # Sources brutes (vue directe sur les tables bronze)
│       └── silver/       # Données nettoyées, dédupliquées, filtrées
├── api/                  # API FastAPI
│   ├── main.py           # Endpoints : /stations, /predict, /predictions, carte
│   └── Dockerfile
├── frontend/             # Carte interactive (HTML/JS/Leaflet), servie par l'API
│   └── index.html
├── notebooks/            # Exploration des données + entraînement du modèle
│   ├── 01_exploration.ipynb
│   └── 02_features_and_baseline_model.ipynb
├── ml_models/            # Modèle entraîné (xgb_baseline.joblib), utilisé par l'API
├── docker-compose.yml    # Orchestration complète (db, adminer, ingestion, api)
├── requirements.txt
└── .env                  # Config (non versionné)
```

## Stack technique

| Composant       | Technologie |
|-----------------|-------------|
| Collecte        | Python (`requests`), boucle continue |
| Stockage        | PostgreSQL 15 |
| Transformation  | dbt |
| Machine Learning| scikit-learn, XGBoost |
| API             | FastAPI, SQLAlchemy |
| Frontend        | HTML / JS vanilla, Leaflet.js (+ MarkerCluster) |
| Orchestration   | Docker Compose |

## Démarrage rapide

### Avec Docker (recommandé — reproduit l'environnement de démonstration)

```bash
docker compose up -d --build
```

Démarre 4 services : `db` (PostgreSQL), `adminer` (interface DB sur :8080), `ingestion` (collecte continue) et `api` (API + carte sur :8000).

Ouvrir **http://localhost:8000/** pour la carte interactive, ou **http://localhost:8000/docs** pour la documentation interactive de l'API.

### En local (développement)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
docker compose up -d db        # uniquement PostgreSQL

python ingestion/collect_velib.py --loop     # collecte
cd dbt_velib/dbt_velib && dbt run            # transformation
uvicorn api.main:app --reload                # API + carte
```

## Endpoints principaux

| Endpoint | Description |
|----------|-------------|
| `GET /` | Carte interactive |
| `GET /health` | Vérification de l'état de l'API |
| `GET /stations` | Liste de toutes les stations avec leur dernier état connu |
| `GET /stations/{id}` | Détail d'une station |
| `GET /stations/{id}/history` | Historique récent d'une station |
| `GET /predict/{id}` | Prédiction pour une station (modèle XGBoost) |
| `GET /predictions` | Prédictions en lot pour toutes les stations |

## Modèle de prévision

Le modèle prédit le nombre de vélos disponibles à la **prochaine collecte**, à partir de :
- l'heure et le jour de la semaine
- les 3 dernières valeurs connues (lags) de la station

Entraîné dans `notebooks/02_features_and_baseline_model.ipynb`, comparé à une baseline de persistance (MAE). À réentraîner régulièrement : la performance s'améliore avec le volume d'historique accumulé par la collecte continue.

## Tests

```bash
pytest tests/
```

## Auteur

Abdelhamid Maaroufi — Master Data / IA / Dev

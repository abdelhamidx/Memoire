"""
API FastAPI qui sert l'état des stations Vélib' (et plus tard les prédictions).

Lancer en local : uvicorn api.main:app --reload
Doc interactive : http://localhost:8000/docs
"""

import json
import logging
import os
from datetime import datetime, timezone

import joblib
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text

from api.auth import (
    CreateUserRequest,
    LoginRequest,
    UpdateUserRequest,
    UserOut,
    authenticate_user,
    create_access_token,
    ensure_auth_tables,
    get_current_user,
    pwd_context,
    require_role,
)
from api.auth import engine as auth_engine

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("velib.api")

DB_URL = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
)
engine = create_engine(DB_URL)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "xgb_baseline.joblib")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé : %s", MODEL_PATH)
except FileNotFoundError:
    model = None
    logger.warning(
        "Modèle introuvable (%s) — /predict, /predictions et /rebalancing renverront 503 "
        "jusqu'à ce qu'il soit entraîné (voir notebooks/02_features_and_baseline_model.ipynb).",
        MODEL_PATH,
    )

N_LAGS = 3

# --- Prédictions multi-horizon (15 / 30 / 60 min) --------------------------
# Voir notebooks/04_train_multi_horizon.ipynb : un modèle est entraîné par
# horizon et comparé à sa propre baseline de persistance. metrics.json dit,
# horizon par horizon, si le modèle appris bat réellement la persistance —
# on ne charge et n'utilise QUE les modèles qui la battent. Un horizon sans
# modèle valable se rabat automatiquement sur la persistance (comportement
# explicite et documenté, pas une approximation silencieuse).
ALLOWED_HORIZONS_MIN = (15, 30, 60)
WEATHER_FEATURE_NAMES = ["temperature_c", "precipitation_mm", "wind_speed_kmh"]

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "metrics.json")
HORIZON_CONFIG: dict[int, dict] = {}
HORIZON_MODELS: dict[int, object] = {}
try:
    with open(METRICS_PATH, encoding="utf-8") as f:
        _metrics = json.load(f)
    for h_str, info in _metrics.get("horizons", {}).items():
        h = int(h_str)
        HORIZON_CONFIG[h] = info
        if info.get("use_model"):
            horizon_model_path = os.path.join(
                os.path.dirname(__file__), "..", "ml_models", f"xgb_h{h}.joblib"
            )
            try:
                HORIZON_MODELS[h] = joblib.load(horizon_model_path)
                logger.info("Modèle horizon %d min chargé : %s", h, horizon_model_path)
            except FileNotFoundError:
                logger.warning(
                    "metrics.json indique use_model=true pour l'horizon %d min mais %s "
                    "est introuvable — repli sur la persistance pour cet horizon.",
                    h, horizon_model_path,
                )
    if HORIZON_CONFIG:
        logger.info(
            "Horizons multi-échéance disponibles : %s (modèle appris pour %s, persistance sinon)",
            sorted(HORIZON_CONFIG), sorted(HORIZON_MODELS) or "aucun",
        )
except FileNotFoundError:
    logger.info(
        "Pas de ml_models/metrics.json — seules les prédictions par défaut (~5 min) sont "
        "disponibles. Voir notebooks/04_train_multi_horizon.ipynb pour générer les horizons "
        "15/30/60 min."
    )


def get_latest_weather(conn) -> dict:
    """Dernier relevé météo connu, utilisé comme feature par les modèles multi-horizon."""
    row = conn.execute(text("""
        select temperature_c, precipitation_mm, wind_speed_kmh
        from dbt_dev.silver_weather
        order by collected_at desc
        limit 1
    """)).mappings().first()
    if row is None:
        return {name: 0.0 for name in WEATHER_FEATURE_NAMES}
    return {name: (row[name] if row[name] is not None else 0.0) for name in WEATHER_FEATURE_NAMES}


def predict_horizon(horizon_minutes: int, now, lags: list[float], weather: dict) -> float:
    """
    Prédit à horizon_minutes avec le modèle appris s'il bat la persistance
    pour cet horizon (voir HORIZON_MODELS / metrics.json), sinon renvoie la
    persistance elle-même (dernière valeur connue) — un choix explicite et
    documenté plutôt qu'un modèle imposé par défaut.
    """
    horizon_model = HORIZON_MODELS.get(horizon_minutes)
    if horizon_model is None:
        return float(lags[0])
    features = [[
        now.hour, now.weekday(),
        lags[0], lags[1], lags[2],
        weather["temperature_c"], weather["precipitation_mm"], weather["wind_speed_kmh"],
    ]]
    return float(horizon_model.predict(features)[0])


# Grille de niveaux d'occupation : sert à colorer la carte et à prioriser
# les stations d'un seul coup d'œil (1 = critique/vide, 5 = saturé/plein).
NIVEAUX = [
    (0.10, 1, "critique", "#ef4444"),
    (0.30, 2, "faible", "#f97316"),
    (0.70, 3, "normal", "#22c55e"),
    (0.90, 4, "eleve", "#38bdf8"),
    (1.01, 5, "sature", "#a855f7"),
]


def niveau_station(bikes, capacity):
    """
    Calcule un niveau d'occupation (1 à 5) à partir du ratio vélos
    disponibles / capacité de la station :
      1 critique (<=10% plein, presque vide, risque de pénurie)
      2 faible   (<=30%)
      3 normal   (30-70%)
      4 élevé    (70-90%)
      5 saturé   (>90%, presque plus de docks libres)
    """
    if not capacity or bikes is None:
        return {"niveau": None, "label": "inconnu", "color": "#64748b"}
    ratio = max(0.0, min(1.0, bikes / capacity))
    for seuil, niveau, label, color in NIVEAUX:
        if ratio <= seuil:
            return {"niveau": niveau, "label": label, "color": color}
    return {"niveau": 5, "label": "sature", "color": "#a855f7"}


app = FastAPI(title="Vélib' Prévision API")

# Sert la carte interactive (frontend/index.html) et ses futurs assets
app.mount("/static", StaticFiles(directory="frontend"), name="static")


def ensure_ops_tables():
    """Table de suivi des actions de rééquilibrage (espace chauffeur)."""
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS ops"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ops.rebalancing_actions (
                id SERIAL PRIMARY KEY,
                station_id TEXT NOT NULL,
                station_name TEXT,
                action_type TEXT NOT NULL,
                suggested_quantity INT,
                agent_name TEXT,
                handled_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        # Notifications envoyées aux chauffeurs quand une nouvelle alerte de
        # rééquilibrage apparaît (voir /rebalancing qui les crée).
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ops.notifications (
                id SERIAL PRIMARY KEY,
                station_id TEXT NOT NULL,
                station_name TEXT,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        # Lecture par utilisateur : chaque chauffeur a son propre statut
        # lu/non-lu sur une même notification.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ops.notification_reads (
                notification_id INT NOT NULL REFERENCES ops.notifications(id),
                user_id INT NOT NULL,
                read_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (notification_id, user_id)
            )
        """))


ensure_ops_tables()
ensure_auth_tables()
logger.info("Démarrage API terminé (tables ops/auth vérifiées).")


@app.post("/auth/login")
def login(payload: LoginRequest):
    """
    Connexion : vérifie l'email/mot de passe et renvoie un token JWT à
    utiliser dans le header `Authorization: Bearer <token>` des appels
    suivants (espace chauffeur, admin).
    """
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")
    token = create_access_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
        },
    }


@app.get("/auth/me")
def read_current_user(current_user: dict = Depends(get_current_user)):
    """Renvoie l'utilisateur connecté (déduit du token JWT)."""
    return current_user


@app.get("/admin/users")
def list_users(_: dict = Depends(require_role("admin"))):
    """Liste tous les comptes (admin uniquement)."""
    with auth_engine.connect() as conn:
        rows = conn.execute(text(
            "select id, email, full_name, role, is_active, created_at from auth.users order by id"
        )).mappings().all()
    return [dict(r) for r in rows]


@app.post("/admin/users", response_model=UserOut)
def create_user(payload: CreateUserRequest, _: dict = Depends(require_role("admin"))):
    """Crée un compte chauffeur ou admin (admin uniquement)."""
    if payload.role not in ("admin", "chauffeur"):
        raise HTTPException(status_code=422, detail="Rôle invalide (admin ou chauffeur)")
    with auth_engine.begin() as conn:
        existing = conn.execute(
            text("select 1 from auth.users where email = :email"), {"email": payload.email}
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="Un compte existe déjà avec cet email")
        row = conn.execute(text("""
            INSERT INTO auth.users (email, password_hash, full_name, role)
            VALUES (:email, :password_hash, :full_name, :role)
            RETURNING id, email, full_name, role, is_active
        """), {
            "email": payload.email,
            "password_hash": pwd_context.hash(payload.password),
            "full_name": payload.full_name,
            "role": payload.role,
        }).mappings().first()
    return dict(row)


@app.patch("/admin/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UpdateUserRequest, _: dict = Depends(require_role("admin"))):
    """Change le rôle, le nom ou active/désactive un compte (admin uniquement)."""
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=422, detail="Aucun champ à mettre à jour")
    if "role" in fields and fields["role"] not in ("admin", "chauffeur"):
        raise HTTPException(status_code=422, detail="Rôle invalide (admin ou chauffeur)")

    set_clause = ", ".join(f"{key} = :{key}" for key in fields)
    with auth_engine.begin() as conn:
        row = conn.execute(
            text(f"""
                UPDATE auth.users SET {set_clause}
                WHERE id = :id
                RETURNING id, email, full_name, role, is_active
            """),
            {**fields, "id": user_id},
        ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return dict(row)


@app.get("/")
def serve_map():
    """Page d'accueil : la carte interactive des stations."""
    return FileResponse("frontend/index.html")


@app.get("/ops")
def serve_ops_dashboard():
    """Espace chauffeur : liste des stations à rééquilibrer en priorité."""
    return FileResponse("frontend/ops.html")


@app.get("/login")
def serve_login_page():
    """Page de connexion (chauffeur ou admin)."""
    return FileResponse("frontend/login.html")


@app.get("/admin")
def serve_admin_dashboard():
    """Panneau d'administration : gestion des comptes chauffeur/admin."""
    return FileResponse("frontend/admin.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/weather/latest")
def weather_latest():
    """
    Dernier relevé météo connu, tel qu'utilisé par les modèles de prédiction
    multi-horizon (voir get_latest_weather / predict_horizon). Sert
    uniquement d'affichage côté frontend — les endpoints de prédiction
    récupèrent cette même donnée en interne, indépendamment de cet appel.
    """
    with engine.connect() as conn:
        row = conn.execute(text("""
            select collected_at, temperature_c, precipitation_mm, wind_speed_kmh
            from dbt_dev.silver_weather
            order by collected_at desc
            limit 1
        """)).mappings().first()
    if row is None:
        return {"available": False}
    return {
        "available": True,
        "collected_at": row["collected_at"],
        "temperature_c": row["temperature_c"],
        "precipitation_mm": row["precipitation_mm"],
        "wind_speed_kmh": row["wind_speed_kmh"],
    }


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
    stations = []
    for row in rows:
        station = dict(row)
        station.update(niveau_station(station["num_bikes_available"], station["capacity"]))
        stations.append(station)
    return stations


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
    station = dict(row)
    station.update(niveau_station(station["num_bikes_available"], station["capacity"]))
    return station


@app.get("/stations/{station_id}/history")
def station_history(station_id: str, limit: int = 50):
    """Historique récent d'une station (pour le mini-graphique de la carte)."""
    query = text("""
        select collected_at, num_bikes_available
        from dbt_dev.silver_station_status
        where station_id = :station_id
        order by collected_at desc
        limit :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"station_id": station_id, "limit": limit}).mappings().all()
    return list(reversed([dict(r) for r in rows]))


@app.get("/predictions")
def predict_all_stations(horizon_minutes: int | None = None):
    """
    Prédiction en lot pour TOUTES les stations en une seule fois (un seul
    appel au modèle), utilisé par la carte pour éviter 1500+ requêtes
    individuelles. Voir /predict/{station_id} pour la version unitaire et
    la doc du paramètre `horizon_minutes`.
    """
    if horizon_minutes is not None and horizon_minutes not in ALLOWED_HORIZONS_MIN:
        raise HTTPException(
            status_code=422,
            detail=f"horizon_minutes doit être l'un de {ALLOWED_HORIZONS_MIN} (ou omis pour ~5 min).",
        )
    if horizon_minutes is None and model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible (ml_models/xgb_baseline.joblib introuvable).",
        )

    query = text("""
        select station_id, collected_at, num_bikes_available
        from (
            select
                station_id, collected_at, num_bikes_available,
                row_number() over (partition by station_id order by collected_at desc) as rn
            from dbt_dev.silver_station_status
        ) ranked
        where rn <= :n_lags
        order by station_id, collected_at desc
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"n_lags": N_LAGS}).mappings().all()
        capacity_rows = conn.execute(text("""
            select station_id, capacity from dbt_dev.silver_station_information
        """)).mappings().all()
        weather = get_latest_weather(conn) if horizon_minutes is not None else None

    capacity_by_station = {r["station_id"]: (r["capacity"] or 0) for r in capacity_rows}

    by_station: dict[str, list] = {}
    for r in rows:
        by_station.setdefault(r["station_id"], []).append(r)

    station_ids, features, currents, based_on, capacities = [], [], [], [], []
    for sid, recs in by_station.items():
        if len(recs) < N_LAGS:
            continue
        recs = sorted(recs, key=lambda r: r["collected_at"], reverse=True)
        now = recs[0]["collected_at"]
        lags = [r["num_bikes_available"] for r in recs[:N_LAGS]]
        station_ids.append(sid)
        features.append([now.hour, now.weekday(), lags[0], lags[1], lags[2]])
        currents.append(lags[0])
        based_on.append(now)
        capacities.append(capacity_by_station.get(sid, 0))

    if not station_ids:
        return []

    if horizon_minutes is None:
        predictions = model.predict(features)
    elif horizon_minutes in HORIZON_MODELS:
        horizon_model = HORIZON_MODELS[horizon_minutes]
        extended = [
            f + [weather["temperature_c"], weather["precipitation_mm"], weather["wind_speed_kmh"]]
            for f in features
        ]
        predictions = horizon_model.predict(extended)
    else:
        # Pas de modèle valable pour cet horizon (voir metrics.json) :
        # on renvoie la persistance (= valeur actuelle) pour chaque station.
        predictions = currents

    def clamp(pred, capacity):
        # Le modèle ignore encore la contrainte physique de capacité :
        # on borne la prédiction dans les limites réelles de la station.
        # Un vélo est une unité entière, donc on arrondit à l'entier le
        # plus proche (pas de décimales comme "13.5 vélos").
        if not capacity:
            return round(max(0.0, float(pred)))
        return round(max(0.0, min(float(pred), capacity)))

    results = []
    for sid, cur, p, bo, cap in zip(station_ids, currents, predictions, based_on, capacities):
        predicted = clamp(p, cap)
        results.append({
            "station_id": sid,
            "current_num_bikes_available": cur,
            "predicted_num_bikes_available": predicted,
            "based_on_collected_at": bo,
            # Niveau calculé sur la PRÉDICTION (pas l'état courant) : c'est
            # ce que la carte doit afficher pour anticiper le rééquilibrage.
            **niveau_station(predicted, cap),
        })
    return results


@app.get("/rebalancing")
def rebalancing_alerts(
    shortage_threshold: int = 2,
    saturation_threshold: int = 2,
    horizon_minutes: int | None = None,
):
    """
    Suggestions de rééquilibrage : stations qui risquent de manquer de vélos
    (pénurie) ou de docks (saturation) à la prochaine collecte (ou à un
    horizon donné, voir /predict/{station_id} pour la doc du paramètre
    `horizon_minutes`), d'après le modèle de prédiction. Utile pour
    prioriser les camions de rééquilibrage.
    """
    if horizon_minutes is not None and horizon_minutes not in ALLOWED_HORIZONS_MIN:
        raise HTTPException(
            status_code=422,
            detail=f"horizon_minutes doit être l'un de {ALLOWED_HORIZONS_MIN} (ou omis pour ~5 min).",
        )
    if horizon_minutes is None and model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible (ml_models/xgb_baseline.joblib introuvable).",
        )

    query = text("""
        select station_id, collected_at, num_bikes_available
        from (
            select
                station_id, collected_at, num_bikes_available,
                row_number() over (partition by station_id order by collected_at desc) as rn
            from dbt_dev.silver_station_status
        ) ranked
        where rn <= :n_lags
        order by station_id, collected_at desc
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"n_lags": N_LAGS}).mappings().all()
        info_rows = conn.execute(text("""
            select station_id, name, capacity, lat, lon
            from dbt_dev.silver_station_information
        """)).mappings().all()
        # Une station traitée par un chauffeur il y a moins de 30 min ne
        # doit pas réapparaître immédiatement dans les alertes.
        recently_handled = conn.execute(text("""
            select distinct station_id
            from ops.rebalancing_actions
            where handled_at > now() - interval '30 minutes'
        """)).scalars().all()
        weather = get_latest_weather(conn) if horizon_minutes is not None else None

    handled_ids = set(recently_handled)
    info_by_station = {r["station_id"]: r for r in info_rows}

    by_station: dict = {}
    for r in rows:
        by_station.setdefault(r["station_id"], []).append(r)

    station_ids, features, currents, meta = [], [], [], []
    for sid, recs in by_station.items():
        if len(recs) < N_LAGS or sid not in info_by_station:
            continue
        # Capacité manquante/à 0 = donnée source corrompue, pas une vraie
        # station : on l'exclut pour éviter les fausses alertes de saturation.
        if not info_by_station[sid]["capacity"]:
            continue
        if sid in handled_ids:
            continue
        recs = sorted(recs, key=lambda r: r["collected_at"], reverse=True)
        now = recs[0]["collected_at"]
        lags = [r["num_bikes_available"] for r in recs[:N_LAGS]]
        station_ids.append(sid)
        features.append([now.hour, now.weekday(), lags[0], lags[1], lags[2]])
        currents.append(lags[0])
        meta.append(info_by_station[sid])

    alerts = []
    if station_ids:
        if horizon_minutes is None:
            predictions = model.predict(features)
        elif horizon_minutes in HORIZON_MODELS:
            horizon_model = HORIZON_MODELS[horizon_minutes]
            extended = [
                f + [weather["temperature_c"], weather["precipitation_mm"], weather["wind_speed_kmh"]]
                for f in features
            ]
            predictions = horizon_model.predict(extended)
        else:
            # Pas de modèle valable pour cet horizon : persistance.
            predictions = currents
        for sid, pred, info in zip(station_ids, predictions, meta):
            capacity = info["capacity"] or 0
            # Le modèle n'a pas encore appris la contrainte physique de
            # capacité (peu de données) : on borne la prédiction dans les
            # limites réelles de la station pour des suggestions réalistes.
            predicted = max(0.0, min(float(pred), capacity))
            predicted_docks = capacity - predicted
            # Cible opérationnelle simple : ramener la station à mi-capacité.
            target = capacity / 2

            if predicted <= shortage_threshold:
                alerts.append({
                    "station_id": sid,
                    "name": info["name"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "type": "penurie",
                    "predicted_num_bikes_available": round(predicted),
                    "capacity": capacity,
                    "severity": round(shortage_threshold - predicted, 1),
                    "suggested_quantity": max(1, round(target - predicted)),
                })
            elif predicted_docks <= saturation_threshold:
                alerts.append({
                    "station_id": sid,
                    "name": info["name"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "type": "saturation",
                    "predicted_num_bikes_available": round(predicted),
                    "capacity": capacity,
                    "severity": round(saturation_threshold - predicted_docks, 1),
                    "suggested_quantity": max(1, round(predicted - target)),
                })

    alerts.sort(key=lambda a: -a["severity"])
    notify_new_alerts(alerts)
    return alerts


def notify_new_alerts(alerts):
    """
    Crée une notification pour chaque alerte qui n'en a pas déjà une
    récente (15 min) sur la même station/type, pour ne pas spammer les
    chauffeurs à chaque rafraîchissement de la page toutes les 30s.
    """
    if not alerts:
        return
    with engine.begin() as conn:
        for a in alerts:
            already_notified = conn.execute(text("""
                select 1 from ops.notifications
                where station_id = :sid and alert_type = :type
                  and created_at > now() - interval '15 minutes'
            """), {"sid": a["station_id"], "type": a["type"]}).first()
            if already_notified:
                continue
            label = "pénurie de vélos" if a["type"] == "penurie" else "saturation (docks pleins)"
            message = (
                f"{a['name']} : {label} prévue "
                f"({a['predicted_num_bikes_available']}/{a['capacity']} vélos)"
            )
            conn.execute(text("""
                INSERT INTO ops.notifications (station_id, station_name, alert_type, message)
                VALUES (:sid, :name, :type, :message)
            """), {
                "sid": a["station_id"],
                "name": a["name"],
                "type": a["type"],
                "message": message,
            })


@app.get("/notifications")
def list_notifications(
    unread_only: bool = False,
    limit: int = 50,
    current_user: dict = Depends(require_role("chauffeur", "admin")),
):
    """Notifications de rééquilibrage, avec statut lu/non-lu propre à l'utilisateur connecté."""
    where_clause = "where r.user_id is null" if unread_only else ""
    query = text(f"""
        select
            n.id, n.station_id, n.station_name, n.alert_type, n.message, n.created_at,
            (r.user_id is not null) as is_read
        from ops.notifications n
        left join ops.notification_reads r
            on r.notification_id = n.id and r.user_id = :user_id
        {where_clause}
        order by n.created_at desc
        limit :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"user_id": current_user["id"], "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


@app.get("/notifications/unread-count")
def unread_notifications_count(current_user: dict = Depends(require_role("chauffeur", "admin"))):
    """Nombre de notifications non lues (pour la pastille sur la cloche)."""
    with engine.connect() as conn:
        count = conn.execute(text("""
            select count(*) from ops.notifications n
            left join ops.notification_reads r
                on r.notification_id = n.id and r.user_id = :user_id
            where r.user_id is null
        """), {"user_id": current_user["id"]}).scalar()
    return {"unread_count": count}


@app.post("/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: int,
    current_user: dict = Depends(require_role("chauffeur", "admin")),
):
    """Marque une notification comme lue pour l'utilisateur connecté."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO ops.notification_reads (notification_id, user_id)
            VALUES (:notification_id, :user_id)
            ON CONFLICT (notification_id, user_id) DO NOTHING
        """), {"notification_id": notification_id, "user_id": current_user["id"]})
    return {"status": "ok"}


@app.post("/rebalancing/{station_id}/handled")
def mark_rebalancing_handled(
    station_id: str,
    action_type: str = "penurie",
    suggested_quantity: int = 0,
    agent_name: str | None = None,
    current_user: dict = Depends(require_role("chauffeur", "admin")),
):
    """
    Espace chauffeur : marque une station comme traitée (vélos chargés ou
    déchargés). L'alerte disparaît de /rebalancing pendant 30 minutes.
    Réservé aux comptes chauffeur/admin connectés.
    """
    # Si le front n'envoie pas de nom explicite, on utilise l'identité du
    # compte connecté (déduite du token JWT).
    if not agent_name:
        agent_name = current_user["email"]

    query = text("""
        select name from dbt_dev.silver_station_information where station_id = :station_id
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"station_id": station_id}).mappings().first()
        station_name = row["name"] if row else station_id

        conn.execute(text("""
            INSERT INTO ops.rebalancing_actions
                (station_id, station_name, action_type, suggested_quantity, agent_name)
            VALUES (:station_id, :station_name, :action_type, :suggested_quantity, :agent_name)
        """), {
            "station_id": station_id,
            "station_name": station_name,
            "action_type": action_type,
            "suggested_quantity": suggested_quantity,
            "agent_name": agent_name,
        })
        conn.commit()

    return {"status": "ok", "station_id": station_id, "station_name": station_name}


@app.get("/rebalancing/history")
def rebalancing_history(limit: int = 50):
    """Historique des actions de rééquilibrage effectuées (espace chauffeur)."""
    query = text("""
        select station_id, station_name, action_type, suggested_quantity, agent_name, handled_at
        from ops.rebalancing_actions
        order by handled_at desc
        limit :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).mappings().all()
    return [dict(r) for r in rows]


@app.get("/predict/{station_id}")
def predict_station(station_id: str, horizon_minutes: int | None = None):
    """
    Prédit le nombre de vélos disponibles pour une station.

    - `horizon_minutes` omis (par défaut) : comportement historique, une
      collecte à l'avance (~5 min) via le modèle XGBoost de
      `notebooks/02_features_and_baseline_model.ipynb`.
    - `horizon_minutes=15|30|60` : utilise le modèle multi-horizon entraîné
      dans `notebooks/04_train_multi_horizon.ipynb` s'il bat la persistance
      pour cet horizon, sinon renvoie la persistance elle-même (voir
      `ml_models/metrics.json`).
    """
    if horizon_minutes is not None and horizon_minutes not in ALLOWED_HORIZONS_MIN:
        raise HTTPException(
            status_code=422,
            detail=f"horizon_minutes doit être l'un de {ALLOWED_HORIZONS_MIN} (ou omis pour ~5 min).",
        )
    if horizon_minutes is None and model is None:
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
        capacity_row = conn.execute(
            text("select capacity from dbt_dev.silver_station_information where station_id = :station_id"),
            {"station_id": station_id},
        ).mappings().first()
        weather = get_latest_weather(conn) if horizon_minutes is not None else None

    if len(rows) < N_LAGS:
        raise HTTPException(
            status_code=422,
            detail=f"Pas assez d'historique pour cette station ({len(rows)}/{N_LAGS} collectes disponibles).",
        )

    now = rows[0]["collected_at"]
    lags = [row["num_bikes_available"] for row in rows]  # [lag_1, lag_2, lag_3]

    if horizon_minutes is None:
        features = [[now.hour, now.weekday(), lags[0], lags[1], lags[2]]]
        prediction = model.predict(features)[0]
        model_name = "xgboost_baseline"
    else:
        prediction = predict_horizon(horizon_minutes, now, lags, weather)
        model_name = f"xgboost_h{horizon_minutes}" if horizon_minutes in HORIZON_MODELS else "persistance"

    capacity = capacity_row["capacity"] if capacity_row else None
    # Le modèle n'a pas encore appris la contrainte physique de capacité
    # (peu de données) : on borne la prédiction dans les limites réelles de
    # la station, comme pour /predictions et /rebalancing.
    # Un vélo est une unité entière : on arrondit à l'entier le plus proche.
    if capacity:
        predicted = round(max(0.0, min(float(prediction), capacity)))
    else:
        predicted = round(max(0.0, float(prediction)))

    return {
        "station_id": station_id,
        "based_on_collected_at": now,
        "current_num_bikes_available": lags[0],
        "predicted_num_bikes_available": predicted,
        "horizon_minutes": horizon_minutes or 5,
        "model": model_name,
    }

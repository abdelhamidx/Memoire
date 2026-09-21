"""
API FastAPI qui servira les prédictions de disponibilité Vélib'.

Pour l'instant : squelette avec un endpoint de santé. À compléter une fois
le modèle entraîné (étape 11-12 du cahier des charges).

Lancer en local : uvicorn api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Vélib' Prévision API")


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: endpoint /predict/{station_id} qui renvoie la disponibilité prédite
# TODO: endpoint /stations qui liste les stations avec leur statut actuel + risque

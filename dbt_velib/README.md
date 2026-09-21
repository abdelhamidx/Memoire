# dbt_velib

Projet dbt pour les transformations bronze → silver → gold.

À initialiser à l'étape 8 du cahier des charges :

```bash
pip install dbt-postgres
cd dbt_velib
dbt init dbt_velib   # ou configurer manuellement dbt_project.yml + profiles.yml
```

Structure prévue (voir le cahier des charges, section "Architecture technique") :
- `models/bronze/` — vues sur les tables brutes (aucune transformation)
- `models/silver/` — nettoyage, typage, dédoublonnage
- `models/gold/` — agrégations, features pour le ML

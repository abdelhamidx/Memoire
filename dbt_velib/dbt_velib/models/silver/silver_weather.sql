-- Une ligne par relevé météo. Pas de déduplication nécessaire (une seule
-- source, pas de risque de doublon comme pour les stations).
select
    collected_at,
    temperature_c,
    precipitation_mm,
    wind_speed_kmh
from {{ ref('bronze_weather') }}
where temperature_c is not null
order by collected_at

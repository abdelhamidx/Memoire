select
    id,
    collected_at,
    temperature_c,
    precipitation_mm,
    wind_speed_kmh
from {{ source('raw', 'weather_raw') }}

select
    station_id,
    name,
    lat,
    lon,
    capacity,
    collected_at
from {{ source('raw', 'station_information_raw') }}
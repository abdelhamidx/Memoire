select distinct on (station_id)
    station_id,
    name,
    lat,
    lon,
    capacity
from {{ ref('bronze_station_information') }}
order by station_id, collected_at desc
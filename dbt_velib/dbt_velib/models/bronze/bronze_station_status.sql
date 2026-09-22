select
    id,
    collected_at,
    station_id,
    num_bikes_available,
    num_docks_available,
    is_installed,
    is_renting,
    is_returning,
    last_reported
from {{ source('raw', 'station_status_raw') }}
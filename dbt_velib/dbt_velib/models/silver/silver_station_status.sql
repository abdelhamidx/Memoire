with deduped as (
    select
        *,
        row_number() over (
            partition by station_id, collected_at
            order by id desc
        ) as rn
    from {{ ref('bronze_station_status') }}
)

select
    station_id,
    collected_at,
    num_bikes_available,
    num_docks_available,
    is_installed,
    is_renting,
    is_returning,
    last_reported
from deduped
where rn = 1
    and is_installed = 1
    and num_bikes_available is not null
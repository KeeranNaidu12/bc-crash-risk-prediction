-- Raw ICBC Reported Crashes (2021-2025), all of BC, loaded as-is.
-- Drop everything built from icbc_crashes first, then the raw table itself.
DROP TABLE IF EXISTS intersection_year;

-- vancouver_crashes may exist as a view (this project) or a table (older version)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_views WHERE viewname = 'vancouver_crashes') THEN
        DROP VIEW vancouver_crashes;
    ELSIF EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'vancouver_crashes') THEN
        DROP TABLE vancouver_crashes;
    END IF;
END $$;

DROP TABLE IF EXISTS icbc_crashes CASCADE;

CREATE TABLE icbc_crashes (
    crash_id                    BIGSERIAL PRIMARY KEY,
    mid_block_crash             TEXT,
    derived_crash_configuration TEXT,
    cyclist_flag                TEXT,
    date_of_loss_year           INTEGER NOT NULL,
    day_of_week                 TEXT,
    heavy_veh_flag              TEXT,
    crash_severity              TEXT,
    time_category               TEXT,
    intersection_crash          TEXT,
    latitude                    DOUBLE PRECISION,
    longitude                   DOUBLE PRECISION,
    month_of_year               TEXT,
    motorcycle_flag             TEXT,
    pedestrian_flag             TEXT,
    animal_flag                 TEXT,
    municipality_with_boundary  TEXT,
    municipality_name           TEXT,
    cross_street_full_name      TEXT,
    parked_vehicle_flag         TEXT,
    parking_lot_flag            TEXT,
    region                      TEXT,
    street_full_name            TEXT,
    road_location_description   TEXT,
    total_victims               INTEGER NOT NULL,
    total_crashes               INTEGER NOT NULL
);
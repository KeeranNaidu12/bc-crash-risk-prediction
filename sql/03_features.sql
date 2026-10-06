-- Static features per Vancouver intersection (model input).
-- Requires intersection_year (02_transform.sql), and intersection_signal and
-- intersection_street_class (written by src/02_load_city_data.py just before this file runs).
-- CASCADE also drops intersection_year's foreign key to this table (re-added below)
DROP TABLE IF EXISTS intersection_features CASCADE;

CREATE TABLE intersection_features AS
WITH locations AS (
    SELECT DISTINCT location, latitude, longitude
    FROM intersection_year
),
-- Streets meeting at the intersection; turning and bus lanes are not streets
streets AS (
    SELECT location, COUNT(*) AS n_streets
    FROM locations, UNNEST(STRING_TO_ARRAY(location, ' & ')) AS street
    WHERE street NOT IN ('TURNING LANE', 'BUS LANE')
    GROUP BY location
)
SELECT l.location,
       l.latitude,
       l.longitude,
       s.n_streets,
       -- Bridges, ramps, highways and connectors (whole word, so CONNAUGHT DR is excluded)
       l.location ~ '(BRIDGE|RAMP|HWY|\mCONN\M)' AS is_interchange,
       g.is_signalized,
       ROUND(g.signal_distance_m::numeric, 1)     AS signal_distance_m,
       -- Most major City street class meeting here (stand-in for traffic volume)
       c.street_class
FROM locations l
JOIN streets s USING (location)
JOIN intersection_signal g USING (location)
JOIN intersection_street_class c USING (location);

ALTER TABLE intersection_features ADD PRIMARY KEY (location);

-- Every intersection-year must belong to a known intersection
ALTER TABLE intersection_year
    ADD FOREIGN KEY (location) REFERENCES intersection_features (location);

-- The hand-off tables from Python are no longer needed; their columns now live here
DROP TABLE intersection_signal;
DROP TABLE intersection_street_class;

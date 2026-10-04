-- Indexes for the filters used below and in later analysis
CREATE INDEX idx_crashes_municipality ON icbc_crashes (municipality_name);
CREATE INDEX idx_crashes_location_year ON icbc_crashes (road_location_description, date_of_loss_year);

-- Vancouver crashes, using the same definition as ICBC's crash maps:
-- no parking-lot crashes and no parked-vehicle crashes.
CREATE VIEW vancouver_crashes AS
SELECT *
FROM icbc_crashes
WHERE municipality_name = 'VANCOUVER'
  AND parking_lot_flag = 'N'
  AND parked_vehicle_flag = 'N';

-- Modelling table: one row per Vancouver intersection per year,
-- including years with zero crashes.
CREATE TABLE intersection_year AS
WITH ints AS (
    SELECT *
    FROM vancouver_crashes
    WHERE intersection_crash = 'Y'
      AND latitude IS NOT NULL
),
locations AS (
    SELECT road_location_description AS location,
           AVG(latitude)  AS latitude,
           AVG(longitude) AS longitude
    FROM ints
    GROUP BY road_location_description
),
years AS (
    SELECT DISTINCT date_of_loss_year AS year FROM ints
),
counts AS (
    SELECT road_location_description AS location,
           date_of_loss_year         AS year,
           SUM(total_crashes)                                                   AS crashes,
           SUM(total_crashes) FILTER (WHERE crash_severity = 'CASUALTY CRASH') AS casualty_crashes,
           SUM(total_victims)                                                   AS victims,
           SUM(total_crashes) FILTER (WHERE pedestrian_flag = 'Y')             AS pedestrian_crashes,
           SUM(total_crashes) FILTER (WHERE cyclist_flag = 'Y')                AS cyclist_crashes,
           SUM(total_crashes) FILTER (WHERE heavy_veh_flag = 'Y')              AS heavy_vehicle_crashes
    FROM ints
    GROUP BY road_location_description, date_of_loss_year
)
SELECT l.location,
       y.year,
       COALESCE(c.crashes, 0)               AS crashes,
       COALESCE(c.casualty_crashes, 0)      AS casualty_crashes,
       COALESCE(c.victims, 0)               AS victims,
       COALESCE(c.pedestrian_crashes, 0)    AS pedestrian_crashes,
       COALESCE(c.cyclist_crashes, 0)       AS cyclist_crashes,
       COALESCE(c.heavy_vehicle_crashes, 0) AS heavy_vehicle_crashes,
       l.latitude,
       l.longitude
FROM locations l
CROSS JOIN years y
LEFT JOIN counts c ON c.location = l.location AND c.year = y.year
ORDER BY l.location, y.year;

ALTER TABLE intersection_year ADD PRIMARY KEY (location, year);
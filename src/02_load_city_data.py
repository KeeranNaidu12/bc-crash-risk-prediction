"""
02_load_city_data.py
Download City of Vancouver traffic signals, flag which crash intersections
are signalized, and build the intersection_features table in PostgreSQL.

Usage (from the project root, after 01_load_icbc.py):
    python src/02_load_city_data.py

Creates in PostgreSQL:
    intersection_features  static features per intersection (see sql/03_features.sql)

The nearest-signal match is computed here in Python and handed to SQL through a
temporary intersection_signal table, which 03_features.sql drops once used.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

from db import get_connection, get_engine, run_sql_file

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"

SIGNALS_URL = ("https://opendata.vancouver.ca/api/explore/v2.1/catalog/"
               "datasets/traffic-signals/exports/geojson")
SIGNALS_FILE = RAW / "traffic-signals.geojson"

# UTM zone 10N: a projection in metres, so distances are accurate in Vancouver
METRES_CRS = "EPSG:26910"
MATCH_DISTANCE_M = 30


def load_signals() -> gpd.GeoDataFrame:
    """Download the signals file once, then reuse the local copy."""
    if not SIGNALS_FILE.exists():
        print("Downloading traffic signals...")
        gpd.read_file(SIGNALS_URL).to_file(SIGNALS_FILE, driver="GeoJSON")
    signals = gpd.read_file(SIGNALS_FILE)
    signals = signals[signals.geometry.notna()].to_crs("EPSG:4326")
    return signals


def main():
    engine = get_engine()

    signals = load_signals()
    print(f"Traffic signals: {len(signals):,}")

    # One point per crash intersection
    ints = pd.read_sql("SELECT DISTINCT location, latitude, longitude FROM intersection_year", engine)
    ints = gpd.GeoDataFrame(ints, geometry=gpd.points_from_xy(ints.longitude, ints.latitude),
                            crs="EPSG:4326").to_crs(METRES_CRS)

    # Distance from each intersection to its nearest signal
    matched = gpd.sjoin_nearest(ints, signals[["geometry"]].to_crs(METRES_CRS),
                                how="left", distance_col="signal_distance_m")
    matched = matched.drop_duplicates("location")  # ties: keep one
    result = matched[["location", "signal_distance_m"]].copy()
    result["is_signalized"] = result["signal_distance_m"] <= MATCH_DISTANCE_M
    result.to_sql("intersection_signal", engine, if_exists="replace", index=False)

    share = result["is_signalized"].mean()
    print(f"Intersections: {len(result):,}  |  signalized (within {MATCH_DISTANCE_M} m): {share:.1%}")

    with get_connection() as conn:
        run_sql_file(conn, SQL_DIR / "03_features.sql")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), SUM(is_interchange::int), AVG(n_streets) "
                        "FROM intersection_features")
            n, interchanges, avg_streets = cur.fetchone()
            print(f"intersection_features: {n:,} rows  |  interchanges: {interchanges}  "
                  f"|  avg streets per intersection: {avg_streets:.2f}")
    conn.close()


if __name__ == "__main__":
    main()
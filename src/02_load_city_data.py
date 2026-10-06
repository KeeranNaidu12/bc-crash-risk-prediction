"""
02_load_city_data.py
Download City of Vancouver traffic signals and street classes, flag which crash
intersections are signalized and how major their streets are, and build the
intersection_features table in PostgreSQL.

Usage (from the project root, after 01_load_icbc.py):
    python src/02_load_city_data.py

Creates in PostgreSQL:
    intersection_features  static features per intersection (see sql/03_features.sql)

The spatial matches are computed here in Python and handed to SQL through temporary
intersection_signal and intersection_street_class tables, which 03_features.sql
drops once used.
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
STREETS_URL = ("https://opendata.vancouver.ca/api/explore/v2.1/catalog/"
               "datasets/public-streets/exports/geojson")
STREETS_FILE = RAW / "public-streets.geojson"

# UTM zone 10N: a projection in metres, so distances are accurate in Vancouver
METRES_CRS = "EPSG:26910"
MATCH_DISTANCE_M = 30
# Street blocks end at the intersection, so the blocks meeting there lie within a few metres
STREET_DISTANCE_M = 15

# City street classes, most to least major. There are no traffic counts per intersection,
# so the class of the busiest street meeting there stands in for traffic volume.
# The few Closed, Recreational and Leased blocks count as local streets.
STREET_CLASSES = ["arterial", "secondary_arterial", "collector", "local"]
STREETUSE_TO_CLASS = {"Arterial": "arterial", "Secondary Arterial": "secondary_arterial",
                      "Collector": "collector"}


def load_signals() -> gpd.GeoDataFrame:
    """Download the signals file once, then reuse the local copy."""
    if not SIGNALS_FILE.exists():
        print("Downloading traffic signals...")
        gpd.read_file(SIGNALS_URL).to_file(SIGNALS_FILE, driver="GeoJSON")
    signals = gpd.read_file(SIGNALS_FILE)
    signals = signals[signals.geometry.notna()].to_crs("EPSG:4326")
    return signals


def load_streets() -> gpd.GeoDataFrame:
    """Download the street blocks file once, then reuse the local copy."""
    if not STREETS_FILE.exists():
        print("Downloading street classes...")
        gpd.read_file(STREETS_URL).to_file(STREETS_FILE, driver="GeoJSON")
    streets = gpd.read_file(STREETS_FILE)
    streets = streets[streets.geometry.notna()].to_crs("EPSG:4326")
    streets["street_class"] = streets["streetuse"].map(STREETUSE_TO_CLASS).fillna("local")
    return streets


def match_street_class(ints: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> pd.DataFrame:
    """Most major class among the street blocks within STREET_DISTANCE_M of each intersection.

    Intersections with no City street block nearby (Stanley Park, Granville Island, the
    port and highway ramps, which the City does not classify) get "none".
    """
    nearby = gpd.sjoin(ints, streets[["street_class", "geometry"]].to_crs(METRES_CRS),
                       predicate="dwithin", distance=STREET_DISTANCE_M)
    nearby["rank"] = nearby["street_class"].map({c: i for i, c in enumerate(STREET_CLASSES)})
    best = nearby.sort_values("rank").drop_duplicates("location")[["location", "street_class"]]
    result = ints[["location"]].merge(best, on="location", how="left")
    result["street_class"] = result["street_class"].fillna("none")
    return result


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

    streets = load_streets()
    print(f"Street blocks: {len(streets):,}")
    street_class = match_street_class(ints, streets)
    street_class.to_sql("intersection_street_class", engine, if_exists="replace", index=False)
    print("Most major street at each intersection:")
    print(street_class["street_class"].value_counts().to_string())

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
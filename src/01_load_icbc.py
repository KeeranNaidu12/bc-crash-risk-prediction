"""
01_load_icbc.py
Read ICBC Reported Crashes (2021-2025) from the Tableau workbook (.twbx)
and load it into PostgreSQL, then build the Vancouver modelling tables.

Usage (from the project root, with the database running):
    python src/01_load_icbc.py

Creates in PostgreSQL:
    icbc_crashes        every crash record in BC, as published
    vancouver_crashes   view: Vancouver, excluding parking-lot and parked-vehicle crashes
    intersection_year   one row per Vancouver intersection per year (model input)
"""
import io
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
from tableauhyperapi import Connection, HyperProcess, TableName, Telemetry

from db import get_connection, run_sql_file

ROOT = Path(__file__).resolve().parents[1]
TWBX = ROOT / "data" / "raw" / "ICBC_Reported_Crashes.twbx"
SQL_DIR = ROOT / "sql"
DATA_FILE = "2021-2025 public data set.hyper"


def read_hyper_from_twbx(twbx_path: Path) -> pd.DataFrame:
    """Extract the crash .hyper file from the workbook and read it into a DataFrame."""
    table = TableName("Extract", "Extract")
    with zipfile.ZipFile(twbx_path) as z, tempfile.TemporaryDirectory() as tmp:
        name = next(n for n in z.namelist() if n.endswith(DATA_FILE))
        hyper_path = z.extract(name, tmp)
        with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
                          parameters={"log_config": ""}) as hyper:
            with Connection(hyper.endpoint, hyper_path) as conn:
                cols = [c.name.unescaped for c in conn.catalog.get_table_definition(table).columns]
                rows = conn.execute_list_query(f"SELECT * FROM {table}")
    df = pd.DataFrame(rows, columns=cols)
    df.columns = df.columns.str.lower()
    return df


def copy_into_postgres(conn, df: pd.DataFrame, table: str):
    """Bulk-load a DataFrame with COPY, which is far faster than row-by-row inserts."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    columns = ", ".join(df.columns)
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv)", buffer)


def main():
    start = time.time()
    df = read_hyper_from_twbx(TWBX)
    print(f"Read {len(df):,} rows from {TWBX.name}")

    with get_connection() as conn:
        run_sql_file(conn, SQL_DIR / "01_schema.sql")
        copy_into_postgres(conn, df, "icbc_crashes")
        print("Loaded icbc_crashes")

        run_sql_file(conn, SQL_DIR / "02_transform.sql")
        print("Built vancouver_crashes and intersection_year")

        with conn.cursor() as cur:
            for table in ["icbc_crashes", "vancouver_crashes", "intersection_year"]:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                print(f"  {table}: {cur.fetchone()[0]:,} rows")
    conn.close()
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
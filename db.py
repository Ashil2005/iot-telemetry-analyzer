"""DuckDB access layer shared by ingest, anomaly detection and the UI.

The decoded Parquet files are the single source of truth; the database file
only holds views over them, so it can be deleted and rebuilt at any time.
"""

from __future__ import annotations

import os

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DECODED_DIR = os.path.join(HERE, "decoded")
DB_PATH = os.path.join(HERE, "telemetry.duckdb")

# Memory guardrail: the whole pipeline must stay under 3 GB, and the decoder
# side of it peaks in the tens of MB, so DuckDB gets a hard 2 GB ceiling.
MEMORY_LIMIT = "2GB"

BASE_VIEW = "telemetry"


def parquet_glob(decoded_dir: str = DECODED_DIR) -> str:
    """Forward-slash glob for read_parquet(), safe to embed in SQL."""
    return os.path.join(decoded_dir, "*.parquet").replace("\\", "/")


def connect(read_only: bool = False, decoded_dir: str = DECODED_DIR,
            db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the database and (re)define the base view over decoded/*.parquet.

    The view is recreated on every connect so the folder can move and newly
    ingested files show up without a rebuild step.
    """
    con = duckdb.connect(db_path, read_only=read_only)
    con.execute("SET memory_limit='%s'" % MEMORY_LIMIT)
    # Timestamps are epoch UTC. Without this DuckDB renders them in the
    # machine's local zone, which silently shifts every displayed time.
    con.execute("SET TimeZone='UTC'")
    if not read_only:
        refresh_view(con, decoded_dir)
    return con


def refresh_view(con: duckdb.DuckDBPyConnection,
                 decoded_dir: str = DECODED_DIR) -> bool:
    """Point the base view at the current contents of ``decoded_dir``.

    Returns False when there is nothing to read yet.
    """
    if not _has_parquet(decoded_dir):
        return False
    # A view body cannot carry bind parameters, so the literal is inlined with
    # quotes doubled.
    pattern = parquet_glob(decoded_dir).replace("'", "''")
    con.execute(
        "CREATE OR REPLACE VIEW %s AS SELECT * FROM read_parquet('%s')"
        % (BASE_VIEW, pattern)
    )
    return True


def _has_parquet(decoded_dir: str) -> bool:
    return os.path.isdir(decoded_dir) and any(
        n.lower().endswith(".parquet") for n in os.listdir(decoded_dir)
    )

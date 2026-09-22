"""Aggregated DuckDB queries backing the Streamlit UI.

Every function here returns something small - a dict or a summary frame.
Nothing in this module ever pulls raw packet rows into Python; the 4M-row
scans all stay inside DuckDB and come back aggregated.

Kept separate from app.py so the data layer can be exercised headlessly.
"""

from __future__ import annotations

import os

import duckdb

import anomalies as an
import db as dbmod

VIEW = an.FLAGGED_VIEW


def data_version(decoded_dir: str = dbmod.DECODED_DIR) -> tuple[int, float]:
    """Cheap fingerprint of decoded/, used as a cache key.

    Changes whenever a file is added or rewritten, so an upload invalidates
    the cached query results without a manual clear.
    """
    if not os.path.isdir(decoded_dir):
        return (0, 0.0)
    files = [os.path.join(decoded_dir, n) for n in os.listdir(decoded_dir)
             if n.endswith(".parquet")]
    if not files:
        return (0, 0.0)
    return (len(files), max(os.path.getmtime(f) for f in files))


def open_connection() -> duckdb.DuckDBPyConnection:
    """Open the database and make sure the flagged view exists."""
    con = dbmod.connect()
    if dbmod.refresh_view(con):
        an.build(con)
    return con


def has_data(decoded_dir: str = dbmod.DECODED_DIR) -> bool:
    return data_version(decoded_dir)[0] > 0


# --------------------------------------------------------------------------
# Tab 1: overview
# --------------------------------------------------------------------------

def headline(con: duckdb.DuckDBPyConnection) -> dict:
    """Single-row corpus summary. One aggregate scan, no row transfer."""
    row = con.cursor().execute("""
        SELECT count(*)                                              AS packets,
               count(DISTINCT source_file)                           AS files,
               100.0 * avg(CASE WHEN sync_ok THEN 1 ELSE 0 END)      AS sync_pct,
               100.0 * avg(CASE WHEN crc_ok  THEN 1 ELSE 0 END)      AS crc_pct,
               CAST(sum(CASE WHEN bad_sync THEN 1 ELSE 0 END) AS BIGINT)       AS bad_sync,
               CAST(sum(CASE WHEN bad_crc THEN 1 ELSE 0 END) AS BIGINT)       AS bad_crc,
               CAST(sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END) AS BIGINT)       AS estops,
               CAST(sum(CASE WHEN vib_spike THEN 1 ELSE 0 END) AS BIGINT)       AS vib_spikes,
               CAST(sum(CASE WHEN temp_spike THEN 1 ELSE 0 END) AS BIGINT)       AS temp_spikes,
               CAST(sum(CASE WHEN ts_reversal THEN 1 ELSE 0 END) AS BIGINT)       AS reversals,
               CAST(sum(CASE WHEN ts_duplicate THEN 1 ELSE 0 END) AS BIGINT)       AS duplicates,
               min(timestamp_seconds)                                AS ts_min,
               max(timestamp_seconds)                                AS ts_max
        FROM %s
    """ % VIEW).fetchone()
    names = ["packets", "files", "sync_pct", "crc_pct", "bad_sync", "bad_crc",
             "estops", "vib_spikes", "temp_spikes", "reversals", "duplicates",
             "ts_min", "ts_max"]
    return dict(zip(names, row))


def per_file_summary(con: duckdb.DuckDBPyConnection):
    """One row per source file. 100 rows out of a 4M-row scan."""
    return con.cursor().execute("""
        SELECT source_file,
               count(*)                                          AS packets,
               100.0 * avg(CASE WHEN sync_ok THEN 1 ELSE 0 END)  AS sync_pct,
               100.0 * avg(CASE WHEN crc_ok  THEN 1 ELSE 0 END)  AS crc_pct,
               CAST(sum(CASE WHEN bad_sync THEN 1 ELSE 0 END) AS BIGINT)   AS bad_sync,
               CAST(sum(CASE WHEN ts_reversal THEN 1 ELSE 0 END) AS BIGINT)   AS reversals,
               CAST(sum(CASE WHEN ts_duplicate THEN 1 ELSE 0 END) AS BIGINT)   AS duplicates,
               CAST(sum(CASE WHEN vib_spike THEN 1 ELSE 0 END) AS BIGINT)   AS vib_spikes,
               CAST(sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END) AS BIGINT)   AS estops,
               round(avg(vibration), 2)                          AS vib_avg,
               max(vibration)                                    AS vib_max,
               -- cast FLOAT -> DOUBLE before rounding, otherwise float32
               -- noise leaks through as 120.400002
               round(CAST(avg(cylinder_temperature) AS DOUBLE), 2) AS temp_avg,
               round(CAST(max(cylinder_temperature) AS DOUBLE), 1) AS temp_max,
               round(avg(motor_rpm), 1)                          AS rpm_avg,
               round(avg(oil_pressure), 2)                       AS psi_avg
        FROM %s
        GROUP BY source_file
        ORDER BY source_file
    """ % VIEW).df()


def source_files(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.cursor().execute(
        "SELECT DISTINCT source_file FROM %s ORDER BY 1" % VIEW).fetchall()
    return [r[0] for r in rows]


def baselines(con: duckdb.DuckDBPyConnection):
    return con.cursor().execute("SELECT * FROM %s" % an.BASELINE_TABLE).df()


# --------------------------------------------------------------------------
# Tab 3: downsampled time series
# --------------------------------------------------------------------------

# Whitelist: only these names may reach the SQL string.
SERIES_CHANNELS: dict[str, str] = {
    "vibration": "counts",
    "cylinder_temperature": "degC",
    "motor_rpm": "RPM",
    "oil_pressure": "PSI",
}

DEFAULT_BUCKETS = 2000


def files_with_estops(con: duckdb.DuckDBPyConnection):
    """Files ranked by emergency-stop count, so the UI can default to one."""
    return con.cursor().execute("""
        SELECT source_file,
               CAST(sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END) AS BIGINT) AS estops
        FROM %s
        GROUP BY source_file
        HAVING sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END) > 0
        ORDER BY estops DESC, source_file
    """ % VIEW).df()


def downsampled_series(con: duckdb.DuckDBPyConnection, source_file: str,
                       channels: list[str], buckets: int = DEFAULT_BUCKETS):
    """Bucket one file's frames to ~``buckets`` points, entirely in DuckDB.

    Returns avg AND max per bucket for every channel: averaging alone would
    erase a single-frame vibration spike of 950 inside a 20-frame bucket, so
    the max series is what keeps anomalies visible after downsampling.
    """
    bad = [c for c in channels if c not in SERIES_CHANNELS]
    if bad:
        raise ValueError("unknown channel(s): %s" % ", ".join(bad))
    if not channels:
        channels = ["vibration"]

    aggregates = ",\n               ".join(
        "avg(CAST(%s AS DOUBLE)) AS %s_avg, max(CAST(%s AS DOUBLE)) AS %s_max"
        % (c, c, c, c) for c in channels
    )
    sql = """
        WITH src AS (
            SELECT frame_index, timestamp_ms, {cols}
            FROM {view}
            WHERE source_file = ?
        ), bucketed AS (
            SELECT NTILE({buckets}) OVER (ORDER BY frame_index) AS bucket, *
            FROM src
        )
        SELECT bucket,
               min(frame_index)                     AS frame_start,
               max(frame_index)                     AS frame_end,
               CAST(avg(frame_index) AS BIGINT)     AS frame_mid,
               count(*)                             AS frames,
               min(timestamp_ms)                    AS ts_start,
               {aggregates}
        FROM bucketed
        GROUP BY bucket
        ORDER BY bucket
    """.format(view=VIEW, buckets=int(buckets), cols=", ".join(channels),
               aggregates=aggregates)
    return con.cursor().execute(sql, [source_file]).df()


def estop_frames(con: duckdb.DuckDBPyConnection, source_file: str) -> list[int]:
    """frame_index of every emergency stop in one file (tens of rows)."""
    rows = con.cursor().execute("""
        SELECT frame_index FROM %s
        WHERE source_file = ? AND emergency_stop
        ORDER BY frame_index
    """ % VIEW, [source_file]).fetchall()
    return [int(r[0]) for r in rows]


# --------------------------------------------------------------------------
# Tab 4: insight
# --------------------------------------------------------------------------

def op_state_escalation(con: duckdb.DuckDBPyConnection):
    """One GROUP BY showing how the channels move across operating states."""
    return con.cursor().execute("""
        SELECT op_state,
               CAST(count(*) AS BIGINT)                            AS rows,
               CAST(sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END)
                    AS BIGINT)                                     AS estops,
               CAST(sum(CASE WHEN vib_spike THEN 1 ELSE 0 END)
                    AS BIGINT)                                     AS vib_spikes,
               round(CAST(avg(vibration) AS DOUBLE), 1)            AS vib_avg,
               round(CAST(max(vibration) AS DOUBLE), 0)            AS vib_max,
               round(CAST(avg(cylinder_temperature) AS DOUBLE), 1) AS temp_avg,
               round(CAST(avg(motor_rpm) AS DOUBLE), 1)            AS rpm_avg,
               round(CAST(avg(oil_pressure) AS DOUBLE), 2)         AS psi_avg
        FROM %s
        GROUP BY op_state
        ORDER BY op_state
    """ % VIEW).df()


def estop_signature(con: duckdb.DuckDBPyConnection) -> dict:
    """Everything the headline block states, computed from the view.

    Nothing here is a constant: the percentages and the enrichment ratio are
    recomputed on whatever data is currently loaded.
    """
    row = con.cursor().execute("""
        WITH e AS (
            SELECT
                CAST(count(*) AS BIGINT)                                AS n_estop,
                CAST(sum(CASE WHEN vib_spike   THEN 1 ELSE 0 END) AS BIGINT) AS vib,
                CAST(sum(CASE WHEN temp_spike  THEN 1 ELSE 0 END) AS BIGINT) AS temp,
                CAST(sum(CASE WHEN state_fault THEN 1 ELSE 0 END) AS BIGINT) AS fault,
                CAST(sum(CASE WHEN rpm_high    THEN 1 ELSE 0 END) AS BIGINT) AS rpm_high,
                CAST(sum(CASE WHEN pressure_drop THEN 1 ELSE 0 END) AS BIGINT) AS psi_drop
            FROM {v} WHERE emergency_stop
        ), n AS (
            SELECT CAST(count(*) AS BIGINT)                          AS n_other,
                   100.0 * avg(CASE WHEN vib_spike THEN 1 ELSE 0 END) AS vib_pct
            FROM {v} WHERE NOT emergency_stop
        ), c AS (
            SELECT corr(vibration, CAST(emergency_stop AS INT))            AS corr_vib,
                   corr(cylinder_temperature, CAST(emergency_stop AS INT)) AS corr_temp,
                   corr(motor_rpm, CAST(emergency_stop AS INT))            AS corr_rpm,
                   corr(oil_pressure, CAST(emergency_stop AS INT))         AS corr_psi
            FROM {v}
        )
        SELECT e.*, n.n_other, n.vib_pct AS other_vib_pct,
               c.corr_vib, c.corr_temp, c.corr_rpm, c.corr_psi
        FROM e, n, c
    """.format(v=VIEW)).fetchone()
    names = ["n_estop", "vib", "temp", "fault", "rpm_high", "psi_drop",
             "n_other", "other_vib_pct", "corr_vib", "corr_temp",
             "corr_rpm", "corr_psi"]
    out = dict(zip(names, row))
    n = out["n_estop"] or 1
    out["vib_pct"] = 100.0 * out["vib"] / n
    out["temp_pct"] = 100.0 * out["temp"] / n
    out["fault_pct"] = 100.0 * out["fault"] / n
    out["enrichment"] = (out["vib_pct"] / out["other_vib_pct"]
                         if out["other_vib_pct"] else float("inf"))
    return out


def file_summary(con: duckdb.DuckDBPyConnection, source_file: str) -> dict:
    """Headline numbers for a single file, used after an import."""
    row = con.cursor().execute("""
        SELECT CAST(count(*) AS BIGINT)                                  AS packets,
               100.0 * avg(CASE WHEN sync_ok THEN 1 ELSE 0 END)          AS sync_pct,
               100.0 * avg(CASE WHEN crc_ok  THEN 1 ELSE 0 END)          AS crc_pct,
               CAST(sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END)
                    AS BIGINT)                                           AS estops,
               CAST(sum(CASE WHEN vib_spike THEN 1 ELSE 0 END)
                    AS BIGINT)                                           AS vib_spikes,
               CAST(sum(CASE WHEN ts_reversal THEN 1 ELSE 0 END)
                    AS BIGINT)                                           AS reversals
        FROM %s WHERE source_file = ?
    """ % VIEW, [source_file]).fetchone()
    return dict(zip(["packets", "sync_pct", "crc_pct", "estops",
                     "vib_spikes", "reversals"], row))


def total_packets(con: duckdb.DuckDBPyConnection) -> int:
    return int(con.cursor().execute(
        "SELECT count(*) FROM %s" % VIEW).fetchone()[0])


def file_frame_count(con: duckdb.DuckDBPyConnection, source_file: str) -> int:
    return int(con.cursor().execute(
        "SELECT count(*) FROM %s WHERE source_file = ?" % VIEW,
        [source_file]).fetchone()[0])

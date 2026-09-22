"""Phase 3: anomaly flags computed in DuckDB over the telemetry view.

Nothing is re-decoded.  Two objects are created in the database:

``baselines``          one row of clean-only summary statistics per channel
``telemetry_flagged``  the base view plus derived z-scores and boolean flags

The baselines are computed over *clean* rows only (``sync_ok AND crc_ok``) so
that injected junk frames and the spikes themselves cannot inflate the mean
or the standard deviation they are then measured against.
"""

from __future__ import annotations

import duckdb

import db as dbmod

FLAGGED_VIEW = "telemetry_flagged"
BASELINE_TABLE = "baselines"

# Rows trusted to define "normal".
CLEAN_PREDICATE = "sync_ok AND crc_ok"

# prefix -> source column.  vibration is the primary spike channel; the others
# are carried so the Phase 4 insight view can correlate against them.
CHANNELS: dict[str, str] = {
    "vib": "vibration",
    "temp": "cylinder_temperature",
    "rpm": "motor_rpm",
    "psi": "oil_pressure",
}

Z_FLAG = 4.0        # z-score threshold for the headline spike flags
MAD_K = 5.0         # robust cross-check: value > median + K * MAD
Z_DROP = -4.0       # one-sided low threshold, used for pressure drops


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def build_baselines(con: duckdb.DuckDBPyConnection,
                    source: str = dbmod.BASE_VIEW) -> dict[str, float]:
    """Compute clean-only mean/stddev/median/MAD for every channel.

    MAD needs the median first, so this runs as two passes: the centre
    statistics, then the median absolute deviation around them.
    """
    centre_cols = ", ".join(
        "avg(%s) AS %s_mean, stddev_samp(%s) AS %s_std, "
        "median(%s) AS %s_median, min(%s) AS %s_min, max(%s) AS %s_max"
        % (col, p, col, p, col, p, col, p, col, p)
        for p, col in CHANNELS.items()
    )
    centre_sql = ("SELECT count(*) AS n_clean, %s FROM %s WHERE %s"
                  % (centre_cols, source, CLEAN_PREDICATE))
    centre = con.execute(centre_sql).fetchone()
    names = [d[0] for d in con.description]
    # median() returns DECIMAL for integer columns; normalise everything to
    # float so callers never mix Decimal and float arithmetic.
    stats = {k: (float(v) if v is not None else None)
             for k, v in zip(names, centre)}

    # Pass 2: MAD about each median.
    mad_cols = ", ".join(
        "median(abs(%s - %r)) AS %s_mad" % (col, float(stats["%s_median" % p]), p)
        for p, col in CHANNELS.items()
    )
    mad_sql = "SELECT %s FROM %s WHERE %s" % (mad_cols, source, CLEAN_PREDICATE)
    mads = con.execute(mad_sql).fetchone()
    stats.update({k: (float(v) if v is not None else None)
                  for k, v in zip([d[0] for d in con.description], mads)})

    # CAST to DOUBLE: a bare decimal literal makes DuckDB infer a DECIMAL type
    # whose scale then overflows once the value is combined arithmetically.
    cols = ", ".join("CAST(%r AS DOUBLE) AS %s" % (float(v) if v is not None else 0.0, k)
                     for k, v in stats.items())
    con.execute("CREATE OR REPLACE TABLE %s AS SELECT %s" % (BASELINE_TABLE, cols))
    return stats


# --------------------------------------------------------------------------
# Flagged view
# --------------------------------------------------------------------------

def _z_expr(prefix: str, col: str) -> str:
    """Null-safe z-score: a zero or null stddev yields NULL, never a crash."""
    return ("(CAST(t.%s AS DOUBLE) - b.%s_mean) / NULLIF(b.%s_std, 0)"
            % (col, prefix, prefix))


def create_flagged_view(con: duckdb.DuckDBPyConnection,
                        source: str = dbmod.BASE_VIEW) -> None:
    """Define ``telemetry_flagged`` over ``source``.

    Every boolean is wrapped in COALESCE so a NULL input (missing column,
    zero variance, first row of a file) becomes ``false`` rather than NULL or
    an error.
    """
    z_cols = ",\n           ".join(
        "%s AS %s_z" % (_z_expr(p, col), p) for p, col in CHANNELS.items()
    )
    mad_flags = ",\n           ".join(
        "COALESCE(CAST(t.%s AS DOUBLE) > b.%s_median + %r * b.%s_mad, false) "
        "AS %s_spike_mad" % (col, p, MAD_K, p, p)
        for p, col in CHANNELS.items()
    )

    sql = """
    CREATE OR REPLACE VIEW {view} AS
    WITH lagged AS (
        SELECT *,
               LAG(timestamp_ms) OVER (
                   PARTITION BY source_file ORDER BY frame_index
               ) AS prev_ts_ms
        FROM {base}
    )
    SELECT t.*,
           -- integrity ---------------------------------------------------
           NOT COALESCE(t.sync_ok, false) AS bad_sync,
           NOT COALESCE(t.crc_ok,  false) AS bad_crc,
           -- timing (first row of each file has prev_ts_ms NULL -> false) -
           COALESCE(t.timestamp_ms <  t.prev_ts_ms, false) AS ts_reversal,
           COALESCE(t.timestamp_ms =  t.prev_ts_ms, false) AS ts_duplicate,
           t.timestamp_ms - t.prev_ts_ms                   AS dt_ms,
           -- z-scores against clean-only baselines ----------------------
           {z_cols},
           -- z-score flags ----------------------------------------------
           COALESCE({vib_z} > {zf}, false) AS vib_spike,
           COALESCE({temp_z} > {zf}, false) AS temp_spike,
           COALESCE({rpm_z} > {zf}, false) AS rpm_high,
           COALESCE({psi_z} < {zd}, false) AS pressure_drop,
           -- robust MAD cross-check -------------------------------------
           {mad_flags},
           -- operational -------------------------------------------------
           COALESCE(t.op_state = 4, false) AS state_fault,
           COALESCE(t.op_state = 3, false) AS state_warning
    FROM lagged t
    CROSS JOIN {baselines} b
    """.format(
        view=FLAGGED_VIEW, base=source, baselines=BASELINE_TABLE,
        z_cols=z_cols, mad_flags=mad_flags, zf=Z_FLAG, zd=Z_DROP,
        vib_z=_z_expr("vib", "vibration"),
        temp_z=_z_expr("temp", "cylinder_temperature"),
        rpm_z=_z_expr("rpm", "motor_rpm"),
        psi_z=_z_expr("psi", "oil_pressure"),
    )

    con.execute(sql)


def build(con: duckdb.DuckDBPyConnection,
          source: str = dbmod.BASE_VIEW) -> dict[str, float]:
    """Compute baselines and (re)create the flagged view."""
    stats = build_baselines(con, source)
    create_flagged_view(con, source)
    return stats


def ensure(con: duckdb.DuckDBPyConnection,
           source: str = dbmod.BASE_VIEW) -> None:
    """Create the flagged view if it is missing or stale."""
    try:
        con.execute("SELECT 1 FROM %s LIMIT 1" % FLAGGED_VIEW).fetchone()
    except duckdb.Error:
        build(con, source)


if __name__ == "__main__":
    connection = dbmod.connect()
    summary = build(connection)
    for key in sorted(summary):
        print("%-14s %s" % (key, summary[key]))
    connection.close()

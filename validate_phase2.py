"""Phase 2 validation: build the DuckDB view and report the corpus numbers.

Usage:  python validate_phase2.py
"""

from __future__ import annotations

import os

import db as dbmod
import decoder as dc
import ingest


def mb(n: float) -> float:
    return n / 1024.0 / 1024.0


def main() -> int:
    print("=" * 78)
    print("PHASE 2 VALIDATION  --  DuckDB view over decoded/*.parquet")
    print("=" * 78)

    con = dbmod.connect()
    print("\n[1] VIEW")
    print("  database      %s" % dbmod.DB_PATH)
    print("  view body     SELECT * FROM read_parquet('%s')" % dbmod.parquet_glob())
    print("  memory_limit  %s" % con.execute(
        "SELECT current_setting('memory_limit')").fetchone()[0])
    cols = con.execute("DESCRIBE %s" % dbmod.BASE_VIEW).fetchall()
    print("  columns       %d" % len(cols))
    for name, ctype, *_ in cols:
        print("      %-28s %s" % (name, ctype))

    # ---- row counts ------------------------------------------------------
    print("\n[2] ROW COUNTS  (expect ~4,000,000)")
    total = con.execute("SELECT count(*) FROM telemetry").fetchone()[0]
    n_src = con.execute(
        "SELECT count(DISTINCT source_file) FROM telemetry").fetchone()[0]
    print("  total rows            %s" % f"{total:,}")
    print("  distinct source_file  %d" % n_src)

    per_file = con.execute("""
        SELECT source_file, count(*) AS rows,
               min(frame_index) AS fi_min, max(frame_index) AS fi_max
        FROM telemetry GROUP BY source_file ORDER BY source_file
    """).fetchall()
    counts = sorted({r[1] for r in per_file})
    print("  per-file rows         min=%d  max=%d  distinct values=%s"
          % (min(c for c in counts), max(c for c in counts), counts))
    print("  sum of per-file rows  %s  (matches total: %s)"
          % (f"{sum(r[1] for r in per_file):,}",
             sum(r[1] for r in per_file) == total))
    bad_idx = [r[0] for r in per_file if (r[2], r[3]) != (0, r[1] - 1)]
    print("  frame_index 0..n-1 in every file: %s"
          % ("YES" if not bad_idx else "NO -> %s" % bad_idx[:5]))
    print("  first 3 / last 3 files:")
    for r in per_file[:3] + per_file[-3:]:
        print("      %-28s rows=%6d  frame_index %d..%d" % (r[0], r[1], r[2], r[3]))

    # ---- size on disk ----------------------------------------------------
    print("\n[3] SIZE ON DISK")
    raw_files = dc.list_raw_files(ingest.RAW_DIR)
    raw_bytes = sum(os.path.getsize(p) for p in raw_files)
    pq_files = [os.path.join(dbmod.DECODED_DIR, n)
                for n in os.listdir(dbmod.DECODED_DIR) if n.endswith(".parquet")]
    pq_bytes = sum(os.path.getsize(p) for p in pq_files)
    db_bytes = os.path.getsize(dbmod.DB_PATH) if os.path.exists(dbmod.DB_PATH) else 0
    print("  raw      %3d files  %9.2f MB" % (len(raw_files), mb(raw_bytes)))
    print("  parquet  %3d files  %9.2f MB   (%.2fx smaller, zstd)"
          % (len(pq_files), mb(pq_bytes), raw_bytes / pq_bytes if pq_bytes else 0))
    print("  duckdb                %9.2f MB   (views only, no copied data)" % mb(db_bytes))
    print("  bytes per decoded packet: raw %.2f  ->  parquet %.2f"
          % (raw_bytes / total, pq_bytes / total))

    # ---- sanity aggregate ------------------------------------------------
    print("\n[4] SANITY QUERY ACROSS ALL FILES")
    row = con.execute("""
        SELECT count(*)                                   AS rows,
               100.0 * avg(CASE WHEN sync_ok THEN 1 ELSE 0 END) AS sync_pct,
               100.0 * avg(CASE WHEN crc_ok  THEN 1 ELSE 0 END) AS crc_pct,
               sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END)  AS estops,
               count(DISTINCT device_id)                  AS n_device_ids,
               min(device_id)                             AS device_id,
               min(timestamp_seconds)                     AS ts_min,
               max(timestamp_seconds)                     AS ts_max
        FROM telemetry
    """).fetchone()
    rows, sync_pct, crc_pct, estops, n_dev, dev, ts_min, ts_max = row
    print("  rows                  %s" % f"{rows:,}")
    print("  sync-pass             %.4f%%   (%s bad)"
          % (sync_pct, f"{round(rows * (100 - sync_pct) / 100):,}"))
    print("  crc-pass              %.4f%%   (%s bad)"
          % (crc_pct, f"{round(rows * (100 - crc_pct) / 100):,}"))
    print("  emergency_stop total  %s" % f"{estops:,}")
    print("  device_id             %d distinct -> 0x%02X" % (n_dev, dev))
    print("  timestamp range       %d .. %d" % (ts_min, ts_max))

    # ---- does it aggregate cleanly per file? -----------------------------
    parts = con.execute("""
        SELECT count(*) AS n_files,
               sum(rows) AS rows, sum(bad_sync) AS bad_sync,
               sum(bad_crc) AS bad_crc, sum(estops) AS estops,
               min(sync_pct) AS min_sync, max(sync_pct) AS max_sync
        FROM (
            SELECT source_file, count(*) AS rows,
                   sum(CASE WHEN NOT sync_ok THEN 1 ELSE 0 END) AS bad_sync,
                   sum(CASE WHEN NOT crc_ok  THEN 1 ELSE 0 END) AS bad_crc,
                   sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END) AS estops,
                   100.0 * avg(CASE WHEN sync_ok THEN 1 ELSE 0 END) AS sync_pct
            FROM telemetry GROUP BY source_file
        )
    """).fetchone()
    n_files, p_rows, p_bad_sync, p_bad_crc, p_estops, min_sync, max_sync = parts
    print("\n  per-file rollup re-aggregated over %d files:" % n_files)
    print("      rows           %s   (equals global: %s)"
          % (f"{p_rows:,}", p_rows == rows))
    print("      bad sync       %s   (equals global: %s)"
          % (f"{p_bad_sync:,}", p_bad_sync == round(rows * (100 - sync_pct) / 100)))
    print("      bad crc        %s" % f"{p_bad_crc:,}")
    print("      emergency_stop %s   (equals global: %s)"
          % (f"{p_estops:,}", p_estops == estops))
    print("      per-file sync%% spread  %.3f%% .. %.3f%%" % (min_sync, max_sync))

    con.close()
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 3 validation: anomaly flag counts, independence checks, insight.

Usage:  python validate_phase3.py
"""

from __future__ import annotations

import anomalies as an
import db as dbmod
import decoder as dc

TOTAL_EXPECTED = 4_000_000


def dcode_label(state: int) -> str:
    return dc.OP_STATE_LABELS.get(int(state), "?")


def rule(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def main() -> int:
    con = dbmod.connect()
    print("=" * 78)
    print("PHASE 3 VALIDATION  --  anomaly flags over %s" % an.FLAGGED_VIEW)
    print("=" * 78)

    stats = an.build(con)
    total = con.execute("SELECT count(*) FROM telemetry").fetchone()[0]

    rule("[1] CLEAN-ONLY BASELINES  (rows with sync_ok AND crc_ok)")
    print("  clean rows used       %s of %s  (%.4f%%)"
          % (f"{int(stats['n_clean']):,}", f"{total:,}",
             100.0 * stats["n_clean"] / total))
    print("  %-22s %9s %9s %9s %9s %9s"
          % ("channel", "mean", "stddev", "median", "MAD", "max"))
    for p, col in an.CHANNELS.items():
        print("  %-22s %9.3f %9.3f %9.3f %9.3f %9.3f"
              % (col, stats["%s_mean" % p], stats["%s_std" % p],
                 stats["%s_median" % p], stats["%s_mad" % p], stats["%s_max" % p]))

    # ---------------------------------------------------------------- flags
    rule("[2] FLAG COUNTS OVER ALL %s ROWS" % f"{total:,}")
    flags = ["bad_sync", "bad_crc", "ts_reversal", "ts_duplicate",
             "vib_spike", "temp_spike", "rpm_high", "pressure_drop",
             "vib_spike_mad", "temp_spike_mad", "rpm_spike_mad", "psi_spike_mad",
             "state_fault", "state_warning", "emergency_stop"]
    sel = ", ".join("sum(CASE WHEN %s THEN 1 ELSE 0 END) AS %s" % (f, f) for f in flags)
    row = con.execute("SELECT %s FROM %s" % (sel, an.FLAGGED_VIEW)).fetchone()
    counts = dict(zip(flags, row))
    print("  %-18s %12s %10s" % ("flag", "count", "% of rows"))
    for f in flags:
        print("  %-18s %12s %9.4f%%" % (f, f"{counts[f]:,}", 100.0 * counts[f] / total))

    print("\n  bad_sync == 20,010 as reported in Phase 2: %s"
          % (counts["bad_sync"] == 20_010))
    marker = con.execute("""
        SELECT count(*) AS n, count(DISTINCT preamble_sync) AS distinct_vals,
               max(preamble_sync) AS val
        FROM %s WHERE bad_sync
    """ % an.FLAGGED_VIEW).fetchone()
    print("  bad_sync rows carry %d distinct preamble value(s) -> 0x%04X (all 0x1234: %s)"
          % (marker[1], marker[2], marker[1] == 1 and marker[2] == 0x1234))

    # -------------------------------------------------- sync/crc independence
    rule("[3] bad_sync x bad_crc  (Phase 1 predicted: independent)")
    grid = con.execute("""
        SELECT bad_sync, bad_crc, count(*) AS n
        FROM %s GROUP BY 1, 2 ORDER BY 1, 2
    """ % an.FLAGGED_VIEW).fetchall()
    cell = {(a, b): n for a, b, n in grid}
    print("                      bad_crc=False   bad_crc=True")
    for s in (False, True):
        print("  bad_sync=%-5s   %15s %14s"
              % (s, f"{cell.get((s, False), 0):,}", f"{cell.get((s, True), 0):,}"))
    both = cell.get((True, True), 0)
    print("\n  overlap (both)        %s" % f"{both:,}")
    print("  -> the 20,010 bad-sync frames ALL pass CRC: %s" % (both == 0))
    print("     CRC cannot detect them; the two flags are genuinely independent.")

    # ------------------------------------------------------------- timestamps
    rule("[4] TIMESTAMP ANOMALIES  (PARTITION BY source_file ORDER BY frame_index)")
    print("  ts_reversal           %12s  %.4f%%"
          % (f"{counts['ts_reversal']:,}", 100.0 * counts["ts_reversal"] / total))
    print("  ts_duplicate          %12s  %.4f%%"
          % (f"{counts['ts_duplicate']:,}", 100.0 * counts["ts_duplicate"] / total))

    nulls = con.execute("""
        SELECT sum(CASE WHEN prev_ts_ms IS NULL THEN 1 ELSE 0 END) AS first_rows,
               sum(CASE WHEN prev_ts_ms IS NULL AND (ts_reversal OR ts_duplicate)
                        THEN 1 ELSE 0 END) AS leaked
        FROM %s
    """ % an.FLAGGED_VIEW).fetchone()
    print("  first row per file (prev_ts_ms NULL)  %d  (== 100 files: %s)"
          % (nulls[0], nulls[0] == 100))
    print("  ...of which wrongly flagged           %d  (NULL -> false, no error: %s)"
          % (nulls[1], nulls[1] == 0))

    # sanity: drop the PARTITION BY and the numbers must change
    unpart = con.execute("""
        WITH g AS (
            SELECT timestamp_ms,
                   LAG(timestamp_ms) OVER (ORDER BY source_file, frame_index) AS prev
            FROM telemetry
        )
        SELECT sum(CASE WHEN timestamp_ms < prev THEN 1 ELSE 0 END) AS rev,
               sum(CASE WHEN timestamp_ms = prev THEN 1 ELSE 0 END) AS dup
        FROM g
    """).fetchone()
    print("\n  SANITY CHECK -- same query without PARTITION BY source_file:")
    print("      reversals  partitioned %s  vs unpartitioned %s   (differs: %s)"
          % (f"{counts['ts_reversal']:,}", f"{unpart[0]:,}",
             counts["ts_reversal"] != unpart[0]))
    print("      duplicates partitioned %s  vs unpartitioned %s   (differs: %s)"
          % (f"{counts['ts_duplicate']:,}", f"{unpart[1]:,}",
             counts["ts_duplicate"] != unpart[1]))
    # Attribute the difference precisely rather than assuming all 99 boundaries
    # go backwards: most files start after the previous one ends.
    boundary = con.execute("""
        WITH edges AS (
            -- first/last BY frame_index, which is what LAG without a
            -- PARTITION actually compares across the boundary. min()/max()
            -- would be the wrong rows, since files contain reversals.
            SELECT source_file,
                   arg_min(timestamp_ms, frame_index) AS first_ms,
                   arg_max(timestamp_ms, frame_index) AS last_ms
            FROM telemetry GROUP BY source_file
        ), seq AS (
            SELECT source_file, first_ms,
                   LAG(last_ms) OVER (ORDER BY source_file) AS prev_last_ms
            FROM edges
        )
        SELECT count(*) FILTER (WHERE prev_last_ms IS NOT NULL) AS boundaries,
               count(*) FILTER (WHERE first_ms < prev_last_ms)  AS backwards,
               count(*) FILTER (WHERE first_ms = prev_last_ms)  AS equal
        FROM seq
    """).fetchone()
    print("      file-to-file boundaries: %d;  crossing backwards: %d;  equal: %d"
          % (boundary[0], boundary[1], boundary[2]))
    print("      boundary reversals (%d) == extra unpartitioned reversals (%d): %s"
          % (boundary[1], unpart[0] - counts["ts_reversal"],
             boundary[1] == unpart[0] - counts["ts_reversal"]))
    print("      -> the %d extra reversal(s) come from boundary crossings, so the"
          % (unpart[0] - counts["ts_reversal"]))
    print("         partitioned flags really are computed within-file.")

    dt = con.execute("""
        SELECT median(dt_ms) AS med, min(dt_ms) AS lo, max(dt_ms) AS hi
        FROM %s WHERE dt_ms IS NOT NULL AND dt_ms > 0
    """ % an.FLAGGED_VIEW).fetchone()
    print("  forward step: median %.0f ms (%.2f Hz), range %d..%d ms"
          % (dt[0], 1000.0 / dt[0], dt[1], dt[2]))

    # ----------------------------------------------------------- spike knee
    rule("[5] SPIKE DETECTION  --  z-score knee vs robust MAD")
    knee = con.execute("""
        SELECT
          sum(CASE WHEN vib_z  > 3 THEN 1 ELSE 0 END) AS v3,
          sum(CASE WHEN vib_z  > 4 THEN 1 ELSE 0 END) AS v4,
          sum(CASE WHEN vib_z  > 5 THEN 1 ELSE 0 END) AS v5,
          sum(CASE WHEN temp_z > 3 THEN 1 ELSE 0 END) AS t3,
          sum(CASE WHEN temp_z > 4 THEN 1 ELSE 0 END) AS t4,
          sum(CASE WHEN temp_z > 5 THEN 1 ELSE 0 END) AS t5
        FROM %s
    """ % an.FLAGGED_VIEW).fetchone()
    print("  %-14s %10s %10s %10s %14s" % ("channel", "z>3", "z>4", "z>5", "MAD (med+5*MAD)"))
    print("  %-14s %10s %10s %10s %14s"
          % ("vibration", f"{knee[0]:,}", f"{knee[1]:,}", f"{knee[2]:,}",
             f"{counts['vib_spike_mad']:,}"))
    print("  %-14s %10s %10s %10s %14s"
          % ("cyl_temp", f"{knee[3]:,}", f"{knee[4]:,}", f"{knee[5]:,}",
             f"{counts['temp_spike_mad']:,}"))
    print("  thresholds: vibration z>4 == value > %.1f | MAD == value > %.1f"
          % (stats["vib_mean"] + 4 * stats["vib_std"],
             stats["vib_median"] + an.MAD_K * stats["vib_mad"]))
    print("              cyl_temp  z>4 == value > %.2f | MAD == value > %.2f"
          % (stats["temp_mean"] + 4 * stats["temp_std"],
             stats["temp_median"] + an.MAD_K * stats["temp_mad"]))

    for label, zcol, madcol in (("vibration", "vib_spike", "vib_spike_mad"),
                                ("cyl_temp", "temp_spike", "temp_spike_mad")):
        ov = con.execute("""
            SELECT sum(CASE WHEN {z} AND {m} THEN 1 ELSE 0 END) AS both,
                   sum(CASE WHEN {z} AND NOT {m} THEN 1 ELSE 0 END) AS z_only,
                   sum(CASE WHEN NOT {z} AND {m} THEN 1 ELSE 0 END) AS mad_only
            FROM {v}
        """.format(z=zcol, m=madcol, v=an.FLAGGED_VIEW)).fetchone()
        print("\n  %s overlap:  both %s | z>4 only %s | MAD only %s"
              % (label, f"{ov[0]:,}", f"{ov[1]:,}", f"{ov[2]:,}"))
        if counts[zcol]:
            print("      every z>4 row is also a MAD row: %s" % (ov[1] == 0))

    # The four spike counts came out identical, which is either a real
    # property of the injected anomalies or a bug. Test it directly.
    same = con.execute("""
        SELECT sum(CASE WHEN vib_spike AND temp_spike THEN 1 ELSE 0 END) AS both,
               sum(CASE WHEN vib_spike AND NOT temp_spike THEN 1 ELSE 0 END) AS vib_only,
               sum(CASE WHEN temp_spike AND NOT vib_spike THEN 1 ELSE 0 END) AS temp_only
        FROM %s
    """ % an.FLAGGED_VIEW).fetchone()
    print("\n  vibration vs temperature spikes: both %s | vib only %s | temp only %s"
          % (f"{same[0]:,}", f"{same[1]:,}", f"{same[2]:,}"))
    print("      -> the two channels spike on the SAME rows: %s"
          % (same[1] == 0 and same[2] == 0))

    sep = con.execute("""
        SELECT min(vibration) AS lo, max(vibration) AS hi,
               min(cylinder_temperature) AS tlo, max(cylinder_temperature) AS thi
        FROM %s WHERE vib_spike
    """ % an.FLAGGED_VIEW).fetchone()
    norm = con.execute("""
        SELECT max(vibration) AS hi, max(cylinder_temperature) AS thi
        FROM %s WHERE NOT vib_spike AND sync_ok
    """ % an.FLAGGED_VIEW).fetchone()
    print("  spike rows:  vibration %d..%d,  temperature %.1f..%.1f"
          % (sep[0], sep[1], sep[2], sep[3]))
    print("  non-spike:   vibration max %d,  temperature max %.1f"
          % (norm[0], norm[1]))
    print("      -> the populations are cleanly separated, which is why z>3, z>4")
    print("         and z>5 all return the same rows: there is nothing in between.")

    rule("[6] EXAMPLE SPIKE ROWS")
    ex = con.execute("""
        SELECT source_file, frame_index, vibration, round(vib_z, 2) AS z,
               round(motor_rpm) AS rpm, oil_pressure,
               round(cylinder_temperature, 1) AS temp_c,
               op_state, emergency_stop
        FROM %s WHERE vib_spike
        ORDER BY vib_z DESC LIMIT 5
    """ % an.FLAGGED_VIEW).fetchall()
    print("  %-26s %7s %6s %7s %7s %5s %7s %4s %6s"
          % ("source_file", "frame", "vib", "z", "rpm", "psi", "temp", "st", "estop"))
    for r in ex:
        print("  %-26s %7d %6d %7.2f %7.0f %5d %7.1f %4d %6s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))

    # -------------------------------------------------------------- insight
    rule("[7] KEY INSIGHT -- what coincides with emergency_stop")
    es = con.execute("""
        SELECT count(*) AS n,
               sum(CASE WHEN vib_spike     THEN 1 ELSE 0 END) AS vib,
               sum(CASE WHEN vib_spike_mad THEN 1 ELSE 0 END) AS vib_mad,
               sum(CASE WHEN temp_spike    THEN 1 ELSE 0 END) AS temp,
               sum(CASE WHEN state_fault   THEN 1 ELSE 0 END) AS fault,
               sum(CASE WHEN rpm_high      THEN 1 ELSE 0 END) AS rpm,
               sum(CASE WHEN pressure_drop THEN 1 ELSE 0 END) AS psi,
               avg(vibration) AS vib_avg, avg(motor_rpm) AS rpm_avg,
               avg(oil_pressure) AS psi_avg, avg(cylinder_temperature) AS temp_avg
        FROM %s WHERE emergency_stop
    """ % an.FLAGGED_VIEW).fetchone()
    n_es = es[0]
    print("  emergency_stop rows: %s" % f"{n_es:,}")
    for label, idx in (("vib_spike (z>4)", 1), ("vib_spike_mad", 2),
                       ("temp_spike (z>4)", 3), ("op_state=4 Fault", 4),
                       ("rpm_high", 5), ("pressure_drop", 6)):
        print("      %-20s %6s  %7.3f%% of e-stops" % (label, f"{es[idx]:,}",
                                                      100.0 * es[idx] / n_es))

    base = con.execute("""
        SELECT 100.0*avg(CASE WHEN vib_spike THEN 1 ELSE 0 END),
               100.0*avg(CASE WHEN vib_spike_mad THEN 1 ELSE 0 END),
               avg(vibration), avg(motor_rpm), avg(oil_pressure),
               avg(cylinder_temperature)
        FROM %s WHERE NOT emergency_stop
    """ % an.FLAGGED_VIEW).fetchone()
    print("\n  channel means:        %12s %12s" % ("e-stop rows", "normal rows"))
    for label, a, b in (("vibration", es[7], base[2]), ("motor_rpm", es[8], base[3]),
                        ("oil_pressure", es[9], base[4]),
                        ("cyl_temp", es[10], base[5])):
        print("      %-16s %12.3f %12.3f   (%+.1f%%)"
              % (label, a, b, 100.0 * (a - b) / b if b else 0.0))
    print("\n  vib_spike rate: %.4f%% on e-stop rows vs %.4f%% elsewhere  -> %.1fx enrichment"
          % (100.0 * es[1] / n_es, base[0],
             (100.0 * es[1] / n_es) / base[0] if base[0] else float("inf")))

    # 15,798 spikes vs 4,017 e-stops: where do the other ~11.8k spikes sit?
    xtab = con.execute("""
        SELECT op_state,
               count(*) AS rows,
               sum(CASE WHEN vib_spike THEN 1 ELSE 0 END) AS spikes,
               sum(CASE WHEN emergency_stop THEN 1 ELSE 0 END) AS estops,
               round(avg(vibration), 1) AS vib_avg,
               round(avg(cylinder_temperature), 1) AS temp_avg,
               round(avg(motor_rpm), 1) AS rpm_avg,
               round(avg(oil_pressure), 2) AS psi_avg
        FROM %s GROUP BY 1 ORDER BY 1
    """ % an.FLAGGED_VIEW).fetchall()
    print("\n  op_state x spike cross-tab:")
    print("  %4s %-8s %12s %10s %9s %8s %9s %8s %8s"
          % ("st", "label", "rows", "spikes", "estops", "vib", "temp", "rpm", "psi"))
    for st, rows_, sp, e, va, ta, ra, pa in xtab:
        print("  %4d %-8s %12s %10s %9s %8.1f %9.1f %8.1f %8.2f"
              % (st, dcode_label(st), f"{rows_:,}", f"{sp:,}", f"{e:,}",
                 va, ta, ra, pa))

    corr = con.execute("""
        SELECT corr(vibration, cylinder_temperature) AS vib_temp,
               corr(vibration, motor_rpm)            AS vib_rpm,
               corr(vibration, oil_pressure)         AS vib_psi,
               corr(cylinder_temperature, motor_rpm) AS temp_rpm
        FROM %s
    """ % an.FLAGGED_VIEW).fetchone()
    print("\n  Pearson correlation over all 4M rows:")
    for label, v in zip(("vibration~cyl_temp", "vibration~motor_rpm",
                         "vibration~oil_pressure", "cyl_temp~motor_rpm"), corr):
        print("      %-24s %+.4f" % (label, v))

    rule("[8] DO SPIKES CLUSTER AROUND e-stop EVENTS?  (offset in frames)")
    prof = con.execute("""
        WITH es AS (
            SELECT source_file, CAST(frame_index AS BIGINT) AS es_idx
            FROM %s WHERE emergency_stop
        )
        SELECT CAST(f.frame_index AS BIGINT) - es.es_idx AS offset,
               count(*) AS n,
               100.0 * avg(CASE WHEN f.vib_spike THEN 1 ELSE 0 END) AS vib_pct,
               avg(f.vibration) AS vib_avg,
               avg(f.motor_rpm) AS rpm_avg,
               avg(f.oil_pressure) AS psi_avg,
               avg(f.cylinder_temperature) AS temp_avg
        FROM %s f
        JOIN es ON f.source_file = es.source_file
               AND CAST(f.frame_index AS BIGINT)
                   BETWEEN es.es_idx - 10 AND es.es_idx + 10
        GROUP BY 1 ORDER BY 1
    """ % (an.FLAGGED_VIEW, an.FLAGGED_VIEW)).fetchall()
    print("  %7s %8s %10s %9s %9s %8s %8s"
          % ("offset", "n", "vib_spike%", "vib_avg", "rpm_avg", "psi_avg", "temp_avg"))
    for off, n, vp, va, ra, pa, ta in prof:
        mark = "  <== e-stop" if off == 0 else ""
        print("  %+7d %8s %9.3f%% %9.2f %9.1f %8.2f %8.2f%s"
              % (off, f"{n:,}", vp, va, ra, pa, ta, mark))

    con.close()
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 1 validation: decode ONE file and report the sanity numbers.

Usage:  python validate_phase1.py [path-to-raw-file]
"""

from __future__ import annotations

import sys
import time

import numpy as np

import decoder as dc


def pct(part: int, whole: int) -> str:
    return "%.4f%%" % (100.0 * part / whole) if whole else "n/a"


def rng(name: str, values: np.ndarray, unit: str = "") -> str:
    v = np.asarray(values, dtype=np.float64)
    return ("  %-22s min=%12.3f  max=%12.3f  mean=%12.3f  %s"
            % (name, v.min(), v.max(), v.mean(), unit))


def main(path: str) -> int:
    print("=" * 74)
    print("PHASE 1 VALIDATION  --  %s" % path)
    print("=" * 74)

    t0 = time.perf_counter()
    frames = dc.read_frames(path)
    n = frames.shape[0]
    file_bytes = n * dc.FRAME_BYTES
    print("\n[1] PACKET COUNT")
    print("  file size             %d bytes" % file_bytes)
    print("  frame size            %d bytes (%d bits)" % (dc.FRAME_BYTES, dc.FRAME_BITS))
    print("  packets decoded       %d   (expected ~40000)" % n)
    print("  trailing bytes        %d" % (np.fromfile(path, dtype=np.uint8).size - file_bytes))

    # ---- bit order check: decode the sync word under both orders ----------
    print("\n[2] SYNC WORD  (expect 0xAA55 on ~99.5%)")
    rates = {}
    for order in ("big", "little"):
        bits = dc.unpack_frames(frames, bitorder=order)
        sync = dc.extract_uint(bits, 0, 16)
        rates[order] = int((sync == dc.SYNC_WORD).sum())
        print("  bitorder=%-7s match %7d / %d   %s"
              % (order, rates[order], n, pct(rates[order], n)))
        del bits, sync

    bitorder = max(rates, key=rates.get)
    print("  --> using bitorder=%r" % bitorder)
    if rates[bitorder] == 0:
        print("  !! NEITHER bit order matches the sync word. Frame alignment is wrong.")
        return 1

    # ---- full decode ------------------------------------------------------
    cols = dc.decode_frames(frames, dc.FULL_SCHEMA, bitorder=bitorder)
    elapsed = time.perf_counter() - t0

    sync_ok = cols["preamble_sync"] == dc.SYNC_WORD
    bad_sync = int((~sync_ok).sum())
    print("  bad sync frames       %d  (%s)" % (bad_sync, pct(bad_sync, n)))

    # ---- device id --------------------------------------------------------
    dev = cols["device_id"]
    uniq, counts = np.unique(dev, return_counts=True)
    print("\n[3] DEVICE_ID  (expect constant 0x4D = 77)")
    print("  distinct values       %d" % uniq.size)
    print("  constant?             %s" % ("YES" if uniq.size == 1 else "NO"))
    top = np.argsort(counts)[::-1][:5]
    for i in top:
        print("      0x%02X (%3d)  x %d  %s" % (uniq[i], uniq[i], counts[i], pct(int(counts[i]), n)))
    dev_ok = dev == dc.DEVICE_ID
    print("  matches 0x4D          %s" % pct(int(dev_ok.sum()), n))

    # ---- timestamps -------------------------------------------------------
    sec = cols["timestamp_seconds"].astype(np.int64)
    ms = cols["sub_second_millis"].astype(np.int64)
    t_ms = sec * 1000 + ms
    d = np.diff(t_ms)
    print("\n[4] TIMESTAMPS  (seconds + sub_second_millis, 14 Hz => ~71 ms step)")
    print("  first epoch s         %d  (%s UTC)"
          % (sec[0], time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(sec[0])))))
    print("  last  epoch s         %d  (%s UTC)"
          % (sec[-1], time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(sec[-1])))))
    print("  span                  %.1f s  (%.2f min)"
          % ((t_ms[-1] - t_ms[0]) / 1000.0, (t_ms[-1] - t_ms[0]) / 60000.0))
    print("  millis range          %d .. %d   (expect 0..999)" % (ms.min(), ms.max()))
    print("  millis > 999          %d" % int((ms > 999).sum()))
    print("  non-decreasing        %s   (%d of %d steps)"
          % (pct(int((d >= 0).sum()), d.size), int((d >= 0).sum()), d.size))
    print("  strictly increasing   %s" % pct(int((d > 0).sum()), d.size))
    print("  reversals (dt < 0)    %d" % int((d < 0).sum()))
    print("  duplicates (dt == 0)  %d" % int((d == 0).sum()))
    pos = d[d > 0]
    if pos.size:
        print("  median forward step   %.1f ms  => %.2f Hz"
              % (np.median(pos), 1000.0 / np.median(pos)))

    # ---- signal ranges ----------------------------------------------------
    print("\n[5] SIGNAL RANGES  -- all frames")
    print(rng("motor_rpm", cols["motor_rpm"], "RPM      (expect 0..12000)"))
    print(rng("cylinder_temperature", cols["cylinder_temperature"], "degC     (expect 0..409.5)"))
    print(rng("vibration", cols["vibration"], "counts   (normal 10..80)"))
    print(rng("oil_pressure", cols["oil_pressure"], "PSI      (expect 0..255)"))

    print("\n    -- sync-valid frames only (%d frames) --" % int(sync_ok.sum()))
    print(rng("motor_rpm", cols["motor_rpm"][sync_ok], "RPM"))
    print(rng("cylinder_temperature", cols["cylinder_temperature"][sync_ok], "degC"))
    print(rng("vibration", cols["vibration"][sync_ok], "counts"))
    print(rng("oil_pressure", cols["oil_pressure"][sync_ok], "PSI"))

    # ---- discrete fields --------------------------------------------------
    print("\n[6] DISCRETE FIELDS")
    st, stc = np.unique(cols["op_state"], return_counts=True)
    for s, c in zip(st, stc):
        print("  op_state %d (%-7s)  x %7d  %s"
              % (s, dc.OP_STATE_LABELS.get(int(s), "?"), c, pct(int(c), n)))
    es = cols["emergency_stop"]
    print("  emergency_stop True    %7d  %s" % (int(es.sum()), pct(int(es.sum()), n)))
    rs = cols["reserved"]
    print("  reserved distinct      %d  (min %d, max %d)" % (np.unique(rs).size, rs.min(), rs.max()))

    # ---- CRC sweep --------------------------------------------------------
    print("\n[7] CRC-16 over bits 0-183 (bytes 0..22), checked against bits 184-199")
    stored = cols["frame_crc"].astype(np.uint16)
    best = None
    for variant in dc.CRC_VARIANTS:
        ok = dc.crc16(frames, variant) == stored
        n_ok = int(ok.sum())
        ok_sync = int((ok & sync_ok).sum())
        flag = ""
        if best is None or n_ok > best[1]:
            best = (variant, n_ok)
        print("  %-14s  all %7d/%d %-10s | sync-valid %7d/%d %s%s"
              % (variant, n_ok, n, pct(n_ok, n), ok_sync, int(sync_ok.sum()),
                 pct(ok_sync, int(sync_ok.sum())), flag))
    print("  --> best variant: %s (%s)" % (best[0], pct(best[1], n)))

    print("\n[8] TIMING:  decoded %d packets in %.2f s  (%.0f k packets/s)"
          % (n, elapsed, n / elapsed / 1000.0))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target is None:
        files = dc.list_raw_files("RAW-DATA")
        if not files:
            print("No .raw/.bin files found in RAW-DATA/")
            raise SystemExit(1)
        target = files[0]
    raise SystemExit(main(target))

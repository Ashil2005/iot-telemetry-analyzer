"""Phase 2: chunked ingest of RAW-DATA/*.raw into decoded/*.parquet.

One file is decoded at a time and written straight to its own Parquet file
through the pyarrow API.  Nothing is ever concatenated across files, so peak
memory is bounded by the largest single file rather than by the corpus.

Usage:  python ingest.py [--force] [--limit N] [--raw-dir DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import sys
import time
from ctypes import wintypes
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import db as dbmod
import decoder as dc

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAW-DATA")
DECODED_DIR = dbmod.DECODED_DIR

CRC_VARIANT = "CCITT-FALSE"   # established in Phase 1: 100% match, XMODEM 0%
BITORDER = "big"              # established in Phase 1: 99.54% vs 0% for little
COMPRESSION = "zstd"

# frame_index and timestamp_ms are monotonic within a file, so delta coding
# beats the default plain/dictionary layout by a wide margin (measured: the
# per-file Parquet drops from 0.795 MB to 0.371 MB).
DELTA_COLUMNS = ("frame_index", "timestamp_ms")

# crc_computed is dropped before writing: it equals frame_crc on every row
# that passes, and crc_ok already records the verdict.  It stays in the
# in-memory decode for diagnostics.
DROP_COLUMNS = ("crc_computed",)

MEMORY_CAP_BYTES = 3 * 1024 ** 3


# --------------------------------------------------------------------------
# Memory probe
# --------------------------------------------------------------------------

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _win32_probe():
    """Bind K32GetProcessMemoryInfo with explicit prototypes.

    Without argtypes/restype ctypes truncates the GetCurrentProcess
    pseudo-handle (-1) to 32 bits and every call silently fails.
    """
    if sys.platform != "win32":
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info = getattr(k32, "K32GetProcessMemoryInfo", None)
        if get_info is None:
            get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE,
                             ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                             wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        return get_info, k32.GetCurrentProcess()
    except (OSError, AttributeError):
        return None


_WIN32_PROBE = _win32_probe()


def rss_now_and_peak() -> tuple[int, int]:
    """(current RSS, peak RSS since process start) in bytes.

    Uses the Win32 working-set counters, which record the true peak rather
    than whatever a sampling loop happens to catch.  Falls back to psutil and
    then to (0, 0) on other platforms.
    """
    if _WIN32_PROBE is not None:
        get_info, handle = _WIN32_PROBE
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if get_info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(info.rss), int(getattr(info, "peak_wset", info.rss))
    except Exception:
        return 0, 0


def mb(n: int) -> float:
    return n / 1024.0 / 1024.0


# --------------------------------------------------------------------------
# Arrow conversion
# --------------------------------------------------------------------------

# Explicit physical types keep the Parquet files compact and stable across
# runs; anything not listed falls back to whatever numpy produced.
ARROW_TYPES: dict[str, pa.DataType] = {
    "preamble_sync": pa.uint16(),
    "device_id": pa.uint8(),
    "timestamp_seconds": pa.uint32(),
    "sub_second_millis": pa.uint16(),
    "timestamp_ms": pa.int64(),
    "motor_rpm": pa.float32(),
    "motor_rpm_raw": pa.uint16(),
    "vibration": pa.uint16(),
    "cylinder_temperature": pa.float32(),
    "cylinder_temperature_raw": pa.uint16(),
    "oil_pressure": pa.uint8(),
    "op_state": pa.uint8(),
    "emergency_stop": pa.bool_(),
    "reserved": pa.uint8(),
    "scrambled_payload": pa.uint64(),
    "frame_crc": pa.uint16(),
    "crc_computed": pa.uint16(),
    "crc_ok": pa.bool_(),
    "sync_ok": pa.bool_(),
    "frame_index": pa.uint32(),
}


def build_table(cols: dict[str, np.ndarray], source_file: str) -> pa.Table:
    """Turn decoded numpy columns into an Arrow table, adding provenance."""
    n = len(next(iter(cols.values())))

    # Derived columns the later phases need, computed once here.
    if "timestamp_seconds" in cols and "sub_second_millis" in cols:
        cols["timestamp_ms"] = (cols["timestamp_seconds"].astype(np.int64) * 1000
                                + cols["sub_second_millis"].astype(np.int64))
    if "preamble_sync" in cols:
        cols["sync_ok"] = cols["preamble_sync"] == dc.SYNC_WORD

    # frame_index preserves the original packet order inside each file, which
    # a glob over Parquet files would otherwise not guarantee.
    cols["frame_index"] = np.arange(n, dtype=np.uint32)

    arrays, names = [], []
    for name, values in cols.items():
        arrays.append(pa.array(values, type=ARROW_TYPES.get(name)))
        names.append(name)

    # source_file is one repeated value: store it dictionary-encoded so it
    # costs a few bytes per file rather than a string per row.
    arrays.append(pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(n, dtype=np.int32)), pa.array([source_file])))
    names.append("source_file")

    return pa.Table.from_arrays(arrays, names=names)


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

def write_parquet(table: pa.Table, out_path: str) -> None:
    """Write one Parquet file with per-column encodings chosen for this data."""
    table = table.drop_columns([c for c in DROP_COLUMNS
                                if c in table.column_names])
    delta = [c for c in DELTA_COLUMNS if c in table.column_names]
    pq.write_table(
        table, out_path,
        compression=COMPRESSION,
        use_dictionary=["source_file"] if "source_file" in table.column_names else False,
        column_encoding={c: "DELTA_BINARY_PACKED" for c in delta},
    )


def ingest_file(path: str, out_dir: str,
                schema: Sequence[dict] = dc.FULL_SCHEMA) -> tuple[str, int]:
    """Decode one raw file and write one Parquet file. Returns (path, rows)."""
    name = os.path.basename(path)
    out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".parquet")

    frames = dc.read_frames(path)
    if frames.shape[0] == 0:
        return out_path, 0

    cols = dc.decode_frames(frames, schema, bitorder=BITORDER,
                            crc_variant=CRC_VARIANT)
    table = build_table(cols, name)
    write_parquet(table, out_path)
    rows = table.num_rows

    # Drop every reference before the next file is touched.
    del table, cols, frames
    gc.collect()
    return out_path, rows


def run(raw_dir: str = RAW_DIR, out_dir: str = DECODED_DIR,
        force: bool = False, limit: int | None = None,
        verbose: bool = True) -> dict:
    """Ingest every raw file one at a time. Returns a report dict."""
    os.makedirs(out_dir, exist_ok=True)
    files = dc.list_raw_files(raw_dir)
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit("No .raw/.bin files found in %s" % raw_dir)

    per_file: list[dict] = []
    raw_bytes = 0
    t0 = time.perf_counter()

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".parquet")
        if not force and os.path.exists(out_path):
            rows = pq.ParquetFile(out_path).metadata.num_rows
            skipped = True
        else:
            _, rows = ingest_file(path, out_dir)
            skipped = False

        cur, peak = rss_now_and_peak()
        raw_bytes += os.path.getsize(path)
        per_file.append({
            "source_file": name,
            "parquet": os.path.basename(out_path),
            "rows": rows,
            "raw_bytes": os.path.getsize(path),
            "parquet_bytes": os.path.getsize(out_path),
            "rss_after": cur,
            "skipped": skipped,
        })
        if verbose and (i % 10 == 0 or i == 1 or i == len(files)):
            print("  [%3d/%d] %-28s rows=%6d  parquet=%6.2f MB  RSS=%6.1f MB  peak=%6.1f MB"
                  % (i, len(files), name, rows,
                     mb(per_file[-1]["parquet_bytes"]), mb(cur), mb(peak)))

    elapsed = time.perf_counter() - t0
    cur, peak = rss_now_and_peak()
    return {
        "files": per_file,
        "n_files": len(files),
        "total_rows": sum(f["rows"] for f in per_file),
        "raw_bytes": raw_bytes,
        "parquet_bytes": sum(f["parquet_bytes"] for f in per_file),
        "elapsed_s": elapsed,
        "rss_final": cur,
        "rss_peak": peak,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-decode files that already have Parquet output")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out", default=DECODED_DIR)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("PHASE 2 INGEST  --  %s -> %s" % (args.raw_dir, args.out))
    print("  crc=%s  bitorder=%s  compression=%s" % (CRC_VARIANT, BITORDER, COMPRESSION))
    print("=" * 78)

    base_cur, base_peak = rss_now_and_peak()
    print("\n[0] baseline RSS before ingest: %.1f MB (peak so far %.1f MB)"
          % (mb(base_cur), mb(base_peak)))

    print("\n[1] DECODING (one file at a time, nothing concatenated)")
    rep = run(args.raw_dir, args.out, force=args.force, limit=args.limit)

    print("\n[2] MEMORY  (rubric: must stay under 3 GB)")
    peak = rep["rss_peak"]
    print("  peak RSS (Win32 PeakWorkingSetSize)   %.1f MB  (%.3f GB)"
          % (mb(peak), peak / 1024.0 ** 3))
    print("  cap                                    3072.0 MB  (3.000 GB)")
    print("  headroom                              %.1f MB   -> using %.2f%% of cap"
          % (mb(MEMORY_CAP_BYTES - peak), 100.0 * peak / MEMORY_CAP_BYTES))
    print("  final RSS after all %d files           %.1f MB"
          % (rep["n_files"], mb(rep["rss_final"])))
    rss_vals = [f["rss_after"] for f in rep["files"]]
    print("  RSS after file 1 / 50 / last          %.1f / %.1f / %.1f MB  (flat => no leak)"
          % (mb(rss_vals[0]), mb(rss_vals[min(49, len(rss_vals) - 1)]), mb(rss_vals[-1])))
    print("  VERDICT: %s" % ("PASS" if peak < MEMORY_CAP_BYTES else "FAIL"))

    print("\n[3] THROUGHPUT")
    print("  %d files, %d rows in %.1f s  (%.0f k packets/s)"
          % (rep["n_files"], rep["total_rows"], rep["elapsed_s"],
             rep["total_rows"] / rep["elapsed_s"] / 1000.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

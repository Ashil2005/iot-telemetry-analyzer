"""Vectorized bit-level decoder for the 200-bit IoT telemetry frame.

The decoder is *schema driven*: every field is a plain dict, so the same code
path serves the built-in specification, the customer-supplied
``dictionary_partial.json`` and whatever the user edits in the Streamlit
data editor.

Frame layout (200 bits = 25 bytes, MSB-first within each byte)::

      0- 15  preamble sync      const 0xAA55
     16- 23  device_id          const 0x4D
     24- 55  timestamp_seconds  uint32 epoch UTC
     56- 65  sub_second_millis  uint 0-999
     66- 77  rpm_raw            uint 0-4095  -> RPM  = raw * 12000/4095
     78- 87  vibration          uint 0-1023
     88- 99  cylinder_temp_raw  uint 0-4095  -> degC = raw * 0.1
    100-107  oil_pressure       uint 0-255 PSI
    108-111  op_state           1=Idle 2=Run 3=Warning 4=Fault
    112      emergency_stop     bool
    113-119  reserved
    120-183  scrambled_payload  XOR-masked (carried through undecoded)
    184-199  frame_crc          CRC-16 over bits 0-183
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

FRAME_BYTES = 25
FRAME_BITS = FRAME_BYTES * 8          # 200
CRC_BIT_OFFSET = 184                  # CRC covers bits 0..183 == bytes 0..22
CRC_COVERED_BYTES = CRC_BIT_OFFSET // 8

SYNC_WORD = 0xAA55
DEVICE_ID = 0x4D

RPM_SCALE = 12000.0 / 4095.0          # 2.930402...
TEMP_SCALE = 0.1

RAW_SUFFIXES = (".raw", ".bin")


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def field(name, bit_offset, bit_length, data_type="uint", *,
          scale_multiplier=1.0, offset_bias=0.0, unit="", notes="",
          expected_constant=None) -> dict:
    """Build one schema entry. Kept as a plain dict so it round-trips JSON."""
    return {
        "name": name,
        "bit_offset": int(bit_offset),
        "bit_length": int(bit_length),
        "data_type": data_type,
        "scale_multiplier": float(scale_multiplier),
        "offset_bias": float(offset_bias),
        "unit": unit,
        "notes": notes,
        "expected_constant": expected_constant,
    }


# The full specification: dictionary_partial.json plus the fields the customer
# never documented.  Phase 4 lets the user edit this interactively.
FULL_SCHEMA: list[dict] = [
    field("preamble_sync", 0, 16, "hex", expected_constant="0xAA55",
          notes="Used to align frame boundaries."),
    field("device_id", 16, 8, "uint", expected_constant="0x4D",
          notes="Undocumented in customer spec; observed constant 77."),
    field("timestamp_seconds", 24, 32, "uint", unit="s",
          notes="Epoch time in seconds (UTC)."),
    field("sub_second_millis", 56, 10, "uint", unit="ms",
          notes="Undocumented; 0-999 sub-second offset."),
    field("motor_rpm", 66, 12, "uint", scale_multiplier=RPM_SCALE, unit="RPM",
          notes="raw 0-4095 mapped onto 0-12000 RPM."),
    field("vibration", 78, 10, "uint", unit="counts",
          notes="Undocumented; 0-1023, normal band 10-80."),
    field("cylinder_temperature", 88, 12, "uint", scale_multiplier=TEMP_SCALE,
          unit="degC", notes="raw 0-4095, 0.1 degC per count."),
    field("oil_pressure", 100, 8, "uint", unit="PSI",
          notes="Undocumented; 0-255 PSI."),
    field("op_state", 108, 4, "uint",
          notes="Undocumented; 1=Idle 2=Run 3=Warning 4=Fault."),
    field("emergency_stop", 112, 1, "bool",
          notes="1 indicates triggered emergency stop."),
    field("reserved", 113, 7, "uint", notes="Undocumented / unused padding."),
    field("scrambled_payload", 120, 64, "uint",
          notes="XOR-masked payload, carried through undecoded."),
    field("frame_crc", 184, 16, "hex", notes="CRC-16 over bits 0-183."),
]

OP_STATE_LABELS = {0: "Unknown", 1: "Idle", 2: "Run", 3: "Warning", 4: "Fault"}


def load_schema(path: str) -> list[dict]:
    """Load a schema from a dictionary JSON file, filling in defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    entries = doc["fields"] if isinstance(doc, dict) else doc
    return [normalize_field(e) for e in entries]


def normalize_field(entry: dict) -> dict:
    """Coerce a loosely-specified field dict into a complete schema entry."""
    return field(
        entry["name"],
        entry["bit_offset"],
        entry["bit_length"],
        entry.get("data_type", "uint"),
        scale_multiplier=entry.get("scale_multiplier", 1.0) or 1.0,
        offset_bias=entry.get("offset_bias", 0.0) or 0.0,
        unit=entry.get("unit", "") or "",
        notes=entry.get("notes", "") or "",
        expected_constant=entry.get("expected_constant"),
    )


def validate_schema(schema: Sequence[dict]) -> list[str]:
    """Return a list of human-readable problems; empty list means usable."""
    problems: list[str] = []
    seen: set[str] = set()
    for entry in schema:
        name = str(entry.get("name", "")).strip()
        if not name:
            problems.append("A field has an empty name.")
            continue
        if name in seen:
            problems.append("Duplicate field name %r." % name)
        seen.add(name)
        try:
            off = int(entry["bit_offset"])
            length = int(entry["bit_length"])
        except (KeyError, TypeError, ValueError):
            problems.append("%s: bit_offset/bit_length must be integers." % name)
            continue
        if length < 1 or length > 64:
            problems.append("%s: bit_length %d outside 1..64." % (name, length))
        elif off < 0 or off + length > FRAME_BITS:
            problems.append(
                "%s: bits %d..%d fall outside the %d-bit frame."
                % (name, off, off + length - 1, FRAME_BITS))
    return problems


# --------------------------------------------------------------------------
# Bit plumbing
# --------------------------------------------------------------------------

def read_frames(path: str, *, frame_bytes: int = FRAME_BYTES) -> np.ndarray:
    """Read a raw file as an ``(n_frames, frame_bytes)`` uint8 array.

    A trailing partial frame is dropped rather than raising.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // frame_bytes
    return raw[: n * frame_bytes].reshape(n, frame_bytes)


def unpack_frames(frames: np.ndarray, *, bitorder: str = "big") -> np.ndarray:
    """``(n, 25)`` bytes -> ``(n, 200)`` bits, MSB-first by default."""
    return np.unpackbits(frames, axis=1, bitorder=bitorder)


def _weights(bit_length: int) -> np.ndarray:
    """MSB-first positional weights, exact up to 64 bits."""
    return np.array([1 << (bit_length - 1 - i) for i in range(bit_length)],
                    dtype=np.uint64)


def extract_uint(bits: np.ndarray, bit_offset: int, bit_length: int) -> np.ndarray:
    """Pack a bit range into uint64 values (one per frame)."""
    window = bits[:, bit_offset: bit_offset + bit_length]
    if bit_length == 1:
        return window[:, 0].astype(np.uint64)
    return window.astype(np.uint64) @ _weights(bit_length)


# --------------------------------------------------------------------------
# CRC
# --------------------------------------------------------------------------

def _crc_table(poly: int, reflected: bool) -> np.ndarray:
    table = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        if reflected:
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
        else:
            crc = i << 8
            for _ in range(8):
                crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table[i] = crc & 0xFFFF
    return table


# name -> (poly, init, reflected, xor_out)
CRC_VARIANTS: dict[str, tuple[int, int, bool, int]] = {
    "CCITT/XMODEM":  (0x1021, 0x0000, False, 0x0000),
    "CCITT-FALSE":   (0x1021, 0xFFFF, False, 0x0000),
    "CCITT-AUG":     (0x1021, 0x1D0F, False, 0x0000),
    "GENIBUS":       (0x1021, 0xFFFF, False, 0xFFFF),
    "KERMIT":        (0x8408, 0x0000, True,  0x0000),
    "MODBUS":        (0xA001, 0xFFFF, True,  0x0000),
    "ARC":           (0xA001, 0x0000, True,  0x0000),
    "USB":           (0xA001, 0xFFFF, True,  0xFFFF),
    "X25":           (0x8408, 0xFFFF, True,  0xFFFF),
}

_TABLE_CACHE: dict[tuple[int, bool], np.ndarray] = {}


def crc16(frames: np.ndarray, variant: str = "CCITT/XMODEM", *,
          n_bytes: int = CRC_COVERED_BYTES) -> np.ndarray:
    """Table-driven CRC-16 over the first ``n_bytes`` of every frame."""
    poly, init, reflected, xor_out = CRC_VARIANTS[variant]
    key = (poly, reflected)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = _crc_table(poly, reflected)
    table = _TABLE_CACHE[key]

    crc = np.full(frames.shape[0], init, dtype=np.uint16)
    data = frames[:, :n_bytes]
    if reflected:
        for j in range(n_bytes):
            idx = ((crc & np.uint16(0xFF)) ^ data[:, j]).astype(np.uint8)
            crc = (crc >> np.uint16(8)) ^ table[idx]
    else:
        for j in range(n_bytes):
            idx = ((crc >> np.uint16(8)) ^ data[:, j]).astype(np.uint8)
            crc = (crc << np.uint16(8)) ^ table[idx]
    return crc ^ np.uint16(xor_out)


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

def decode_frames(frames: np.ndarray, schema: Sequence[dict] = FULL_SCHEMA, *,
                  bitorder: str = "big",
                  crc_variant: str | None = "CCITT/XMODEM") -> dict[str, np.ndarray]:
    """Decode an ``(n, 25)`` uint8 frame array into a dict of columns.

    Scaled fields also emit a ``<name>_raw`` column so the untouched counts
    stay available for recalibration.
    """
    bits = unpack_frames(frames, bitorder=bitorder)
    out: dict[str, np.ndarray] = {}

    for entry in schema:
        name = entry["name"]
        offset, length = int(entry["bit_offset"]), int(entry["bit_length"])
        if offset < 0 or length < 1 or offset + length > bits.shape[1]:
            continue  # validate_schema() reports this; never crash here
        raw = extract_uint(bits, offset, length)
        dtype = str(entry.get("data_type", "uint")).lower()

        if dtype == "bool":
            out[name] = raw.astype(bool)
            continue

        scale = float(entry.get("scale_multiplier", 1.0) or 1.0)
        bias = float(entry.get("offset_bias", 0.0) or 0.0)
        if dtype in ("hex", "raw") or (scale == 1.0 and bias == 0.0):
            out[name] = _downcast(raw, length)
        else:
            out[name] = raw.astype(np.float64) * scale + bias
            out[name + "_raw"] = _downcast(raw, length)

    if crc_variant:
        computed = crc16(frames, crc_variant)
        out["crc_computed"] = computed
        stored = out.get("frame_crc")
        if stored is not None:
            out["crc_ok"] = np.asarray(stored).astype(np.uint16) == computed

    return out


def _downcast(values: np.ndarray, bit_length: int) -> np.ndarray:
    """Shrink a uint64 column to the narrowest exact unsigned dtype."""
    if bit_length <= 8:
        return values.astype(np.uint8)
    if bit_length <= 16:
        return values.astype(np.uint16)
    if bit_length <= 32:
        return values.astype(np.uint32)
    return values


def decode_file(path: str, schema: Sequence[dict] = FULL_SCHEMA, *,
                bitorder: str = "big",
                crc_variant: str | None = "CCITT/XMODEM",
                max_frames: int | None = None) -> dict[str, np.ndarray]:
    """Read and decode one raw file."""
    frames = read_frames(path)
    if max_frames is not None:
        frames = frames[:max_frames]
    return decode_frames(frames, schema, bitorder=bitorder,
                         crc_variant=crc_variant)


def list_raw_files(directory: str = "RAW-DATA") -> list[str]:
    """Every .raw/.bin file in ``directory``, sorted by name."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, n) for n in os.listdir(directory)
        if n.lower().endswith(RAW_SUFFIXES)
    )

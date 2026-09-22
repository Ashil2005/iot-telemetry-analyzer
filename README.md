# IoT Telemetry Decoder & Analyzer

Decodes binary industrial telemetry — 200-bit frames at 14 Hz — and analyses it
end to end: a vectorized bit-level decoder, a memory-bounded ingest to Parquet,
anomaly detection in DuckDB, and a Streamlit UI over 4,000,000 packets.

Built against a corpus of 100 × 1 MB `.raw` files (40,000 packets each). The data
is not in this repository; drop your own `RAW-DATA/` in and run the ingest.

---

## Quickstart

```bash
git clone <your-repo-url>
cd <repo>

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# put your .raw / .bin files in RAW-DATA/, then:
python ingest.py                  # 100 files -> decoded/*.parquet  (~19 s)
python anomalies.py               # baselines + flagged view

streamlit run app.py
```

Requires Python 3.12+.

---

## Frame format

200 bits = 25 bytes per packet, MSB-first within each byte.

| Bits | Field | Notes |
|---|---|---|
| 0–15 | `preamble_sync` | constant `0xAA55` |
| 16–23 | `device_id` | constant `0x4D` (77) |
| 24–55 | `timestamp_seconds` | uint32, epoch UTC |
| 56–65 | `sub_second_millis` | 0–999 |
| 66–77 | `motor_rpm` | raw 0–4095 → `raw × 12000/4095` RPM |
| 78–87 | `vibration` | 0–1023 counts |
| 88–99 | `cylinder_temperature` | raw 0–4095 → `raw × 0.1` °C |
| 100–107 | `oil_pressure` | 0–255 PSI |
| 108–111 | `op_state` | 1=Idle 2=Run 3=Warning 4=Fault |
| 112 | `emergency_stop` | bool |
| 113–119 | `reserved` | all-zero in this corpus |
| 120–183 | `scrambled_payload` | constant across all 4M packets |
| 184–199 | `frame_crc` | CRC-16 over bits 0–183 |

The supplied `dictionary_partial.json` documents only 5 of these 13 fields. The
rest were recovered by decoding; the app's schema editor lets you re-derive them
interactively.

### Three things the spec got wrong

- **The CRC is CCITT-FALSE, not XMODEM.** Same polynomial `0x1021`, but
  `init=0xFFFF`. Swept against 9 standard variants: CCITT-FALSE matches
  **4,000,000 / 4,000,000**; XMODEM matches **0**. Coverage confirmed as exactly
  bits 0–183 (22 bytes → 0 matches, 23 → all, 24 → 0).
- **Bad-sync frames are injected markers, not corruption.** All 20,010 carry the
  identical value `0x1234`, keep `device_id = 0x4D`, and **pass CRC**. Sync and
  CRC are independent checks — the 2×2 overlap is exactly zero.
- **RPM and oil pressure carry no fault signal.** Both are smooth cyclic
  carriers. Correlation with `emergency_stop` is −0.0004 and +0.0006, and
  neither channel can reach 4σ (they peak at 1.51σ and 1.54σ).

---

## What the analysis found

**Every one of the 4,017 emergency stops** carries a vibration spike, a
temperature spike and `op_state = 4` simultaneously. Vibration spikes occur on
100% of e-stop rows versus 0.2948% elsewhere — a **339× enrichment**.

| op_state | packets | e-stops | avg vibration | avg temp | avg RPM | avg PSI |
|---|---|---|---|---|---|---|
| 2 Run | 3,984,202 | 0 | 30.0 | 78.6 | 3198.6 | 44.50 |
| 3 Warning | 11,781 | 0 | 577.1 | 118.6 | 3197.8 | 44.52 |
| 4 Fault | 4,017 | 4,017 | 874.7 | 118.6 | 3195.2 | 44.57 |

Temperature jumps once and saturates; **vibration magnitude is what separates a
warning from a fault**. Spikes are isolated single-frame events — the ±10-frame
window around each e-stop sits at the background rate, so nothing in this data
predicts a fault in advance.

---

## Architecture

| Module | Role |
|---|---|
| `decoder.py` | Vectorized bit-slicing (`unpackbits` → dot with powers of two), schema-driven, CRC-16 table-driven |
| `ingest.py` | One file at a time → one Parquet each; Win32 peak-RSS probe |
| `db.py` | DuckDB connection + view over `decoded/*.parquet` |
| `anomalies.py` | Clean-only baselines + `telemetry_flagged` view (all flags in SQL) |
| `queries.py` | Aggregated queries for the UI — nothing pulls raw rows into Python |
| `app.py` | Streamlit UI only; no decode logic duplicated |
| `validate_phase1/2/3.py` | Runnable evidence for each stage |

**The decoder is schema-driven throughout.** A field is a plain dict, so the
built-in specification, `dictionary_partial.json` and whatever you type into the
UI all flow through one code path.

### Memory discipline

Nothing is ever concatenated across files. Measured peaks against a 3 GB budget:

| Stage | Peak RSS | % of 3 GB |
|---|---|---|
| Ingest (100 files, 4M packets) | 132 MB | 4.3% |
| Anomaly detection | 855 MB | 27.8% |
| Full UI render, all 5 tabs | 1,286 MB | 41.9% |

RSS after files 1 / 50 / 100 is 101.3 / 102.7 / 101.9 MB — flat, no accumulation.
Charts downsample **inside DuckDB** (NTILE, 40,000 → 2,000 at 20:1), carrying both
average *and* max per bucket so single-frame spikes survive: vibration peaks at
950 in the max series but only 112 in the average.

Parquet output is 37.17 MB against 95.37 MB of raw input (2.57× smaller) using
zstd plus delta encoding on the monotonic columns.

---

## The five tabs

| Tab | Shows |
|---|---|
| **Overview** | Corpus headline + per-file summary, fully aggregated in DuckDB |
| **Schema editor** | Editable 13-field dictionary, live bit-coverage, validation that blocks bad input, "Test on 1 File" re-decoding 40k packets in ~54 ms |
| **Time series** | Any file/channels, NTILE-downsampled, average line with the average→peak band |
| **Insight** | The fault signature, in σ units against the clean baselines, with the escalation table |
| **Import** | Upload one `.raw`/`.bin` with size and duplicate guards, then a single-file ingest |

---

## Validation

Each stage ships a script that prints its own evidence:

```bash
python validate_phase1.py    # decode one file: counts, sync rate, CRC sweep, ranges
python validate_phase2.py    # corpus: row counts, sizes, per-file vs global rollups
python validate_phase3.py    # flags, independence checks, spike knee, insight
```

Phase 1 cross-checks the vectorized decoder against an independent scalar
implementation (`int.from_bytes` + shifts): 504 frames × 11 fields, zero
mismatches. Phase 2 verifies the Parquet round-trip against a fresh in-memory
decode: 12 columns × 40,000 rows, zero mismatches.

"""IoT telemetry decoder & analyzer - Streamlit UI.

Run with:  .venv\\Scripts\\streamlit run app.py

This module is a UI only.  All decoding lives in decoder.py, all ingest in
ingest.py, all SQL in db.py / anomalies.py / queries.py.
"""

from __future__ import annotations

import datetime as dt_mod
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import anomalies as an
import db as dbmod
import decoder as dc
import ingest
import queries as q

st.set_page_config(page_title="IoT Telemetry Analyzer",
                   page_icon="chart_with_upwards_trend",
                   layout="wide")


# --------------------------------------------------------------------------
# Shared resources
# --------------------------------------------------------------------------

@st.cache_resource
def get_connection():
    """One DuckDB connection for the session; queries use .cursor() on it."""
    return q.open_connection()


@st.cache_data(show_spinner=False)
def cached_headline(version):
    return q.headline(get_connection())


@st.cache_data(show_spinner=False)
def cached_per_file(version):
    return q.per_file_summary(get_connection())


@st.cache_data(show_spinner=False)
def cached_source_files(version):
    return q.source_files(get_connection())


def fmt_int(n) -> str:
    return f"{int(n):,}"


# --------------------------------------------------------------------------
# Tab 1: overview
# --------------------------------------------------------------------------

def render_overview() -> None:
    st.subheader("Corpus overview")

    version = q.data_version()
    head = cached_headline(version)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total packets", fmt_int(head["packets"]))
    c2.metric("Files", fmt_int(head["files"]))
    c3.metric("Sync pass", "%.4f%%" % head["sync_pct"],
              delta="-%s bad" % fmt_int(head["bad_sync"]),
              delta_color="inverse")
    c4.metric("CRC pass", "%.4f%%" % head["crc_pct"],
              delta=("-%s bad" % fmt_int(head["bad_crc"])
                     if head["bad_crc"] else "all verify"),
              delta_color="inverse" if head["bad_crc"] else "off")
    c5.metric("Emergency stops", fmt_int(head["estops"]))
    c6.metric("Vibration spikes", fmt_int(head["vib_spikes"]))

    st.caption(
        "CRC-16/CCITT-FALSE over bits 0-183 - %s of %s packets verify. "
        "The %s bad-sync frames carry 0x1234 and still pass CRC, "
        "so the two checks are independent."
        % (fmt_int(head["packets"] - head["bad_crc"]), fmt_int(head["packets"]),
           fmt_int(head["bad_sync"]))
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Timestamp reversals", fmt_int(head["reversals"]))
    d2.metric("Timestamp duplicates", fmt_int(head["duplicates"]))
    d3.metric("Temperature spikes", fmt_int(head["temp_spikes"]))
    span_h = (head["ts_max"] - head["ts_min"]) / 3600.0
    d4.metric("Time span", "%.1f h" % span_h)

    st.divider()
    st.subheader("Per-file summary")
    df = cached_per_file(version)
    st.caption(
        "%d files aggregated inside DuckDB - %s packets scanned, %d rows returned."
        % (len(df), fmt_int(head["packets"]), len(df))
    )
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "source_file": st.column_config.TextColumn("file", width="medium"),
            "packets": st.column_config.NumberColumn("packets", format="%d"),
            "sync_pct": st.column_config.NumberColumn("sync %", format="%.3f"),
            "crc_pct": st.column_config.NumberColumn("crc %", format="%.3f"),
            "bad_sync": st.column_config.NumberColumn("bad sync", format="%d"),
            "reversals": st.column_config.NumberColumn("ts rev", format="%d"),
            "duplicates": st.column_config.NumberColumn("ts dup", format="%d"),
            "vib_spikes": st.column_config.NumberColumn("vib spikes", format="%d"),
            "estops": st.column_config.NumberColumn("e-stops", format="%d"),
            "vib_avg": st.column_config.NumberColumn("vib avg", format="%.2f"),
            "vib_max": st.column_config.NumberColumn("vib max", format="%d"),
            "temp_avg": st.column_config.NumberColumn("temp avg", format="%.2f"),
            "temp_max": st.column_config.NumberColumn("temp max", format="%.1f"),
            "rpm_avg": st.column_config.NumberColumn("rpm avg", format="%.1f"),
            "psi_avg": st.column_config.NumberColumn("psi avg", format="%.2f"),
        },
    )

    with st.expander("Clean-only baselines used for spike detection"):
        st.caption(
            "Computed over rows with sync_ok AND crc_ok so injected junk and "
            "the spikes themselves cannot inflate the mean or stddev."
        )
        st.dataframe(q.baselines(get_connection()), width="stretch",
                     hide_index=True)


# --------------------------------------------------------------------------
# Tab 2: schema editor
# --------------------------------------------------------------------------

SCHEMA_COLUMNS = ["name", "bit_offset", "bit_length", "data_type",
                  "scale_multiplier"]
DATA_TYPES = ["uint", "bool", "hex", "raw"]
PARTIAL_DICT = os.path.join(dbmod.HERE, "dictionary_partial.json")


def schema_to_rows(schema) -> list[dict]:
    """Schema dicts -> the five editable columns."""
    return [{k: f.get(k) for k in SCHEMA_COLUMNS} for f in schema]


def rows_to_records(frame: pd.DataFrame) -> list[dict]:
    """Editor grid -> plain dicts, with blanks normalised to None.

    Values are passed through untouched so validate_schema() sees exactly
    what the user typed and can report on it, rather than crashing here.
    """
    records = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                clean[key] = None
            elif isinstance(value, str):
                clean[key] = value.strip()
            else:
                clean[key] = value
        records.append(clean)
    return records


def describe_columns(cols: dict) -> pd.DataFrame:
    """Per-field min/max/mean over the decoded columns."""
    rows = []
    for name, values in cols.items():
        arr = np.asarray(values)
        numeric = arr.astype(np.float64) if arr.dtype == bool else arr
        rows.append({
            "field": name,
            "dtype": str(arr.dtype),
            "min": float(np.min(numeric)),
            "max": float(np.max(numeric)),
            "mean": float(np.mean(numeric)),
        })
    return pd.DataFrame(rows)


def render_schema_editor() -> None:
    st.subheader("Editable frame dictionary")
    st.caption(
        "The customer dictionary documents 5 of 13 fields. The grid is seeded "
        "with the full layout recovered during decoding; edit, add or delete "
        "rows and re-decode a single file to check the result."
    )

    if "schema_rows" not in st.session_state:
        st.session_state.schema_rows = schema_to_rows(dc.FULL_SCHEMA)

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Reset to full recovered schema", width="stretch"):
        st.session_state.schema_rows = schema_to_rows(dc.FULL_SCHEMA)
        st.rerun()
    if b2.button("Reset to partial dictionary", width="stretch"):
        st.session_state.schema_rows = schema_to_rows(dc.load_schema(PARTIAL_DICT))
        st.rerun()

    edited = st.data_editor(
        pd.DataFrame(st.session_state.schema_rows, columns=SCHEMA_COLUMNS),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="schema_editor",
        column_config={
            "name": st.column_config.TextColumn("name", required=True),
            "bit_offset": st.column_config.NumberColumn(
                "bit_offset", min_value=0, max_value=dc.FRAME_BITS - 1, step=1),
            "bit_length": st.column_config.NumberColumn(
                "bit_length", min_value=1, max_value=64, step=1),
            "data_type": st.column_config.SelectboxColumn(
                "data_type", options=DATA_TYPES),
            "scale_multiplier": st.column_config.NumberColumn(
                "scale_multiplier", format="%.6f"),
        },
    )

    records = rows_to_records(edited)
    problems = dc.validate_schema(records)

    if problems:
        st.error("Schema has %d problem(s) - fix before decoding:" % len(problems))
        for p in problems:
            st.write("- " + p)
    else:
        covered = sum(int(r["bit_length"]) for r in records)
        st.success(
            "Schema valid: %d fields covering %d of %d frame bits (%.1f%%)."
            % (len(records), covered, dc.FRAME_BITS,
               100.0 * covered / dc.FRAME_BITS)
        )

    st.divider()
    st.subheader("Test on 1 file")
    st.caption(
        "Decodes a single file in memory with the edited schema. "
        "Nothing is written to decoded/ and the corpus is not reprocessed."
    )

    files = cached_source_files(q.data_version())
    p1, p2, p3 = st.columns([3, 2, 2])
    picked = p1.selectbox("Source file", files, index=0, key="schema_test_file")
    limit = p2.number_input("Max frames (0 = all)", min_value=0, max_value=40000,
                            value=0, step=1000, key="schema_test_limit")
    run = p3.button("Test on 1 File", type="primary", width="stretch",
                    disabled=bool(problems))

    if problems:
        p3.caption("Disabled while the schema has problems.")

    if run:
        raw_path = os.path.join(ingest.RAW_DIR, picked)
        if not os.path.exists(raw_path):
            st.error("Raw file not found: %s" % raw_path)
            return

        schema = [dc.normalize_field(r) for r in records]
        t0 = time.perf_counter()
        frames = dc.read_frames(raw_path)
        if limit:
            frames = frames[:int(limit)]
        cols = dc.decode_frames(frames, schema, bitorder=ingest.BITORDER,
                                crc_variant=ingest.CRC_VARIANT)
        elapsed = time.perf_counter() - t0

        m1, m2, m3 = st.columns(3)
        m1.metric("Packets decoded", fmt_int(frames.shape[0]))
        m2.metric("Decode time", "%.3f s" % elapsed)
        m3.metric("Throughput", "%s pkt/s"
                  % fmt_int(frames.shape[0] / elapsed if elapsed else 0))

        st.markdown("**head(20)**")
        head = pd.DataFrame({k: np.asarray(v)[:20] for k, v in cols.items()})
        st.dataframe(head, width="stretch", hide_index=True)

        st.markdown("**Per-field min / max / mean**")
        st.dataframe(describe_columns(cols), width="stretch", hide_index=True,
                     column_config={
                         "min": st.column_config.NumberColumn(format="%.3f"),
                         "max": st.column_config.NumberColumn(format="%.3f"),
                         "mean": st.column_config.NumberColumn(format="%.3f"),
                     })

        del cols, frames


# --------------------------------------------------------------------------
# Tab 3: downsampled time series
# --------------------------------------------------------------------------

# Categorical slots 1 and 2 from the validated reference palette, plus the
# reserved critical status colour for emergency stops.
SERIES_1 = "#2a78d6"
SERIES_1_FILL = "rgba(42, 120, 214, 0.16)"
SERIES_2 = "#eb6834"
SERIES_2_FILL = "rgba(235, 104, 52, 0.16)"
STATUS_CRITICAL = "#e34948"
GRID = "rgba(128, 128, 128, 0.18)"

CHANNEL_LABELS = {
    "vibration": "Vibration (counts)",
    "cylinder_temperature": "Cylinder temperature (degC)",
    "motor_rpm": "Motor speed (RPM)",
    "oil_pressure": "Oil pressure (PSI)",
}


@st.cache_data(show_spinner=False)
def cached_estop_files(version):
    return q.files_with_estops(get_connection())


@st.cache_data(show_spinner=False)
def cached_series(version, source_file, channels, buckets):
    return q.downsampled_series(get_connection(), source_file, list(channels),
                                buckets)


@st.cache_data(show_spinner=False)
def cached_estops(version, source_file):
    return q.estop_frames(get_connection(), source_file)


@st.cache_data(show_spinner=False)
def cached_frame_count(version, source_file):
    return q.file_frame_count(get_connection(), source_file)


def add_channel_panel(fig, df, channel, row, colour, fill, show_legend, x=None):
    """One channel: average line with the avg->max band drawn behind it.

    The band is what keeps a single-frame spike of 950 visible after a 20:1
    downsample; the average alone flattens it to ~75.

    ``x`` defaults to the per-file frame axis; the range view passes a
    timestamp axis instead so a window spanning files plots continuously.
    """
    if x is None:
        x = df["frame_mid"]
    fig.add_trace(
        go.Scatter(x=x, y=df[channel + "_avg"], name="bucket average",
                   legendgroup="avg", showlegend=show_legend,
                   line=dict(color=colour, width=2),
                   hovertemplate="%{x}<br>avg %{y:.2f}<extra></extra>"),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=df[channel + "_max"], name="bucket peak (max)",
                   legendgroup="max", showlegend=show_legend,
                   line=dict(color=colour, width=1),
                   opacity=0.5, fill="tonexty", fillcolor=fill,
                   hovertemplate="%{x}<br>max %{y:.2f}<extra></extra>"),
        row=row, col=1,
    )


def mark_estops(fig, estops, n_rows, show_legend=True, limit=250):
    """Vertical rules at every emergency stop, plus one legend proxy."""
    for frame in estops[:limit]:
        fig.add_vline(x=frame, line=dict(color=STATUS_CRITICAL, width=1),
                      opacity=0.35, row="all", col=1)
    if show_legend and estops:
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", showlegend=True,
                       name="emergency stop (%d)" % len(estops),
                       line=dict(color=STATUS_CRITICAL, width=1)),
            row=1, col=1,
        )


def style_figure(fig, n_rows, height_per=190):
    fig.update_layout(
        height=height_per * n_rows + 90,
        hovermode="x unified",
        margin=dict(l=8, r=8, t=64, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_xaxes(title_text="frame_index (packet order within file)",
                     row=n_rows, col=1)


def render_time_series() -> None:
    st.subheader("Time series")

    version = q.data_version()
    estop_files = cached_estop_files(version)
    files = cached_source_files(version)
    # Default to the file with the most emergency stops so the view opens on
    # something worth looking at.
    default = (estop_files.iloc[0]["source_file"]
               if len(estop_files) else files[0])

    c1, c2, c3 = st.columns([2, 3, 1])
    picked = c1.selectbox("Source file", files, index=files.index(default),
                          key="ts_file")
    channels = c2.multiselect("Channels", list(q.SERIES_CHANNELS),
                              default=list(q.SERIES_CHANNELS), key="ts_channels")
    buckets = c3.number_input("Buckets", min_value=200, max_value=5000,
                              value=q.DEFAULT_BUCKETS, step=100, key="ts_buckets")

    if not channels:
        st.info("Select at least one channel.")
        return

    df = cached_series(version, picked, tuple(channels), int(buckets))
    estops = cached_estops(version, picked)
    raw_rows = cached_frame_count(version, picked)
    ratio = raw_rows / len(df) if len(df) else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw frames in file", fmt_int(raw_rows))
    m2.metric("Points plotted", fmt_int(len(df)))
    m3.metric("Downsample", "%s -> %s, %.0f:1"
              % (fmt_int(raw_rows), fmt_int(len(df)), ratio))
    m4.metric("Emergency stops", fmt_int(len(estops)))

    st.caption(
        "NTILE(%d) bucketing runs inside DuckDB; only %d aggregated rows reach "
        "the browser. Each panel shows the bucket average as a line with the "
        "average-to-peak band behind it - the peak series is what keeps a "
        "single-frame spike visible at %.0f:1."
        % (int(buckets), len(df), ratio)
    )

    fig = make_subplots(
        rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[CHANNEL_LABELS.get(c, c) for c in channels],
    )
    for i, channel in enumerate(channels, start=1):
        add_channel_panel(fig, df, channel, i, SERIES_1, SERIES_1_FILL,
                          show_legend=(i == 1))
    mark_estops(fig, estops, len(channels))
    style_figure(fig, len(channels))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Downsampled data (what the chart actually receives)"):
        st.dataframe(df.head(50), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Range analysis: the whole corpus as one timeline
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_span(version):
    return q.corpus_span(get_connection())


@st.cache_data(show_spinner=False)
def cached_range_summary(version, t0, t1):
    return q.range_summary(get_connection(), t0, t1)


@st.cache_data(show_spinner=False)
def cached_range_files(version, t0, t1):
    return q.range_files(get_connection(), t0, t1)


@st.cache_data(show_spinner=False)
def cached_range_series(version, t0, t1, channels, buckets):
    return q.range_series(get_connection(), t0, t1, list(channels), buckets)


@st.cache_data(show_spinner=False)
def cached_range_estops(version, t0, t1, limit):
    return q.range_estop_times(get_connection(), t0, t1, limit)


def to_utc(epoch: int) -> dt_mod.datetime:
    return dt_mod.datetime.fromtimestamp(int(epoch), dt_mod.timezone.utc)


def from_parts(day: dt_mod.date, clock: dt_mod.time) -> int:
    """Combine a date and a time as UTC and return epoch seconds."""
    return int(dt_mod.datetime.combine(day, clock,
                                       tzinfo=dt_mod.timezone.utc).timestamp())


def humanise(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return "%dd %dh %dm" % (days, hours, minutes)
    if hours:
        return "%dh %dm" % (hours, minutes)
    return "%dm %ds" % (minutes, secs)


# Vertical rules stop being readable past this many events; beyond it the
# per-bucket count panel carries the information instead.
VLINE_BUDGET = 200


def render_range_analysis() -> None:
    st.subheader("Date / time range across the whole corpus")

    version = q.data_version()
    span = cached_span(version)
    t_min, t_max = int(span["t_min"]), int(span["t_max"])

    st.caption(
        "All %d files are loaded as one continuous timeline: %s packets from "
        "%d device(s), %s UTC to %s UTC (%.1f hours, no gaps). A range below "
        "is analysed across file boundaries."
        % (span["files"], fmt_int(span["rows"]), span["devices"],
           to_utc(t_min).strftime("%Y-%m-%d %H:%M:%S"),
           to_utc(t_max).strftime("%Y-%m-%d %H:%M:%S"), span["span_hours"])
    )

    presets = {
        "Full span (default)": (t_min, t_max),
        "First 6 hours": (t_min, min(t_min + 6 * 3600, t_max)),
        "Last 6 hours": (max(t_max - 6 * 3600, t_min), t_max),
        "First 24 hours": (t_min, min(t_min + 24 * 3600, t_max)),
        "Custom": None,
    }
    def apply_window(a: int, b: int) -> None:
        """Push a window into the four date/time widgets.

        Written before the widgets are instantiated this run, which is the
        only point at which a keyed widget will accept a new value - setting
        it afterwards is ignored and the preset appears to do nothing.
        """
        lo_, hi_ = to_utc(a), to_utc(b)
        st.session_state["range_d0"] = lo_.date()
        st.session_state["range_t0"] = lo_.time()
        st.session_state["range_d1"] = hi_.date()
        st.session_state["range_t1"] = hi_.time()

    if "range_d0" not in st.session_state:
        apply_window(t_min, t_max)

    previous = st.session_state.get("_range_preset_prev", "Full span (default)")
    choice = st.selectbox("Preset", list(presets), index=0, key="range_preset")
    if choice != previous:
        st.session_state["_range_preset_prev"] = choice
        if presets[choice] is not None:
            apply_window(*presets[choice])

    c1, c2, c3, c4 = st.columns(4)
    start_day = c1.date_input("Start date (UTC)", key="range_d0",
                              min_value=to_utc(t_min).date(),
                              max_value=to_utc(t_max).date())
    start_clock = c2.time_input("Start time (UTC)", key="range_t0", step=1)
    end_day = c3.date_input("End date (UTC)", key="range_d1",
                            min_value=to_utc(t_min).date(),
                            max_value=to_utc(t_max).date())
    end_clock = c4.time_input("End time (UTC)", key="range_t1", step=1)

    t0 = from_parts(start_day, start_clock)
    t1 = from_parts(end_day, end_clock)

    if t0 >= t1:
        st.error("The start of the window must be earlier than its end.")
        return

    clipped = max(t0, t_min), min(t1, t_max)
    if (t0, t1) != clipped:
        st.info("Window clipped to the corpus extent.")
        t0, t1 = clipped

    summary = cached_range_summary(version, t0, t1)
    if not summary["rows"]:
        st.warning("No packets fall inside this window.")
        return

    st.markdown(
        "**Selected window:** %s -> %s UTC  |  duration **%s**"
        % (to_utc(t0).strftime("%Y-%m-%d %H:%M:%S"),
           to_utc(t1).strftime("%Y-%m-%d %H:%M:%S"), humanise(t1 - t0))
    )

    # ---- headline tiles for the slice --------------------------------
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Packets in window", fmt_int(summary["rows"]),
              delta="%.1f%% of corpus" % (100.0 * summary["rows"] / span["rows"]),
              delta_color="off")
    m2.metric("Files spanned", fmt_int(summary["files"]),
              delta="of %d" % span["files"], delta_color="off")
    m3.metric("Sync pass", "%.4f%%" % summary["sync_pct"],
              delta="-%s bad" % fmt_int(summary["bad_sync"]),
              delta_color="inverse")
    m4.metric("CRC pass", "%.4f%%" % summary["crc_pct"],
              delta=("-%s bad" % fmt_int(summary["bad_crc"])
                     if summary["bad_crc"] else "all verify"),
              delta_color="inverse" if summary["bad_crc"] else "off")
    m5.metric("Emergency stops", fmt_int(summary["estops"]))
    m6.metric("Vibration spikes", fmt_int(summary["vib_spikes"]))

    d1_, d2_, d3_, d4_ = st.columns(4)
    d1_.metric("Timestamp reversals", fmt_int(summary["reversals"]))
    d2_.metric("Timestamp duplicates", fmt_int(summary["duplicates"]))
    d3_.metric("Temperature spikes", fmt_int(summary["temp_spikes"]))
    d4_.metric("Actual data span",
               humanise(summary["last_ts"] - summary["first_ts"]))

    st.caption(
        "Filtered over the full telemetry_flagged view, so ts_reversal and "
        "ts_duplicate still compare each packet with the one that genuinely "
        "preceded it - including across a file boundary inside the window."
    )

    ch1, ch2 = st.columns(2)
    ch1.metric("Vibration avg / max", "%.2f / %s"
               % (summary["vib_avg"], fmt_int(summary["vib_max"])))
    ch2.metric("Cyl temp avg / max", "%.2f / %.1f"
               % (summary["temp_avg"], summary["temp_max"]))
    ch3, ch4 = st.columns(2)
    ch3.metric("Motor rpm avg / max", "%.1f / %.1f"
               % (summary["rpm_avg"], summary["rpm_max"]))
    ch4.metric("Oil pressure avg / max", "%.2f / %s"
               % (summary["psi_avg"], fmt_int(summary["psi_max"])))

    # ---- which files the window touches ------------------------------
    files_df = cached_range_files(version, t0, t1)
    with st.expander("Files this window spans (%d)" % len(files_df),
                     expanded=len(files_df) <= 12):
        shown = files_df.copy()
        shown["from_utc"] = [to_utc(v).strftime("%Y-%m-%d %H:%M:%S")
                             for v in shown["first_ts"]]
        shown["to_utc"] = [to_utc(v).strftime("%Y-%m-%d %H:%M:%S")
                           for v in shown["last_ts"]]
        st.dataframe(
            shown[["source_file", "rows", "from_utc", "to_utc", "estops"]],
            width="stretch", hide_index=True,
            column_config={
                "source_file": st.column_config.TextColumn("file"),
                "rows": st.column_config.NumberColumn("packets", format="%d"),
                "from_utc": st.column_config.TextColumn("from (UTC)"),
                "to_utc": st.column_config.TextColumn("to (UTC)"),
                "estops": st.column_config.NumberColumn("e-stops", format="%d"),
            },
        )
        partial = files_df[files_df["rows"] < 40000]
        if len(partial):
            st.caption(
                "%d file(s) are only partially inside the window "
                "(%s) - the range cuts into files, not just between them."
                % (len(partial), ", ".join(
                    "%s: %s rows" % (r.source_file, fmt_int(r.rows))
                    for r in partial.itertuples())[:400])
            )

    st.divider()

    # ---- downsampled chart over the window ---------------------------
    s1, s2 = st.columns([3, 1])
    channels = s1.multiselect("Channels", list(q.SERIES_CHANNELS),
                              default=list(q.SERIES_CHANNELS),
                              key="range_channels")
    buckets = s2.number_input("Buckets", min_value=200, max_value=5000,
                              value=q.DEFAULT_BUCKETS, step=100,
                              key="range_buckets")
    if not channels:
        st.info("Select at least one channel.")
        return

    series = cached_range_series(version, t0, t1, tuple(channels), int(buckets))
    ratio = summary["rows"] / max(len(series), 1)
    x = pd.to_datetime(series["ts_mid"], unit="ms", utc=True)

    st.caption(
        "%s packets bucketed to %s points in DuckDB (%.0f:1), ordered by "
        "timestamp rather than frame index so the series stays continuous "
        "across the %d file(s). Each bucket carries its peak, so single-frame "
        "spikes survive."
        % (fmt_int(summary["rows"]), fmt_int(len(series)), ratio,
           summary["files"])
    )

    n_rows = len(channels) + 1
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=[CHANNEL_LABELS.get(c, c) for c in channels]
                       + ["Emergency stops per bucket"],
    )
    for i, channel in enumerate(channels, start=1):
        add_channel_panel(fig, series, channel, i, SERIES_1, SERIES_1_FILL,
                          show_legend=(i == 1), x=x)

    fig.add_trace(
        go.Bar(x=x, y=series["estops"], name="emergency stops",
               marker=dict(color=STATUS_CRITICAL),
               hovertemplate="%{x}<br>%{y} e-stop(s)<extra></extra>"),
        row=n_rows, col=1,
    )

    estop_times = cached_range_estops(version, t0, t1, VLINE_BUDGET + 1)
    if 0 < len(estop_times) <= VLINE_BUDGET:
        mark_estops(fig, [pd.Timestamp(v, unit="ms", tz="UTC")
                          for v in estop_times], n_rows)
    elif summary["estops"]:
        st.caption(
            "%s emergency stops fall in this window - too many to mark "
            "individually, so the bottom panel counts them per bucket. Narrow "
            "the window below %d to see individual markers."
            % (fmt_int(summary["estops"]), VLINE_BUDGET)
        )

    style_figure(fig, n_rows, height_per=170)
    fig.update_xaxes(title_text="timestamp (UTC)", row=n_rows, col=1)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Downsampled data (what the chart receives)"):
        st.dataframe(series.head(50), width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# Tab 4: insight
# --------------------------------------------------------------------------

# channel column -> baseline prefix used by anomalies.py
CHANNEL_PREFIX = {col: prefix for prefix, col in an.CHANNELS.items()}

SPIKE_PANEL = ["vibration", "cylinder_temperature"]
CARRIER_PANEL = ["motor_rpm", "oil_pressure"]


@st.cache_data(show_spinner=False)
def cached_escalation(version):
    return q.op_state_escalation(get_connection())


@st.cache_data(show_spinner=False)
def cached_signature(version):
    return q.estop_signature(get_connection())


@st.cache_data(show_spinner=False)
def cached_baselines(version):
    return q.baselines(get_connection())


def peak_sigma(base, prefix: str) -> float:
    """Largest absolute z-score a channel actually reaches in the corpus."""
    mean = float(base["%s_mean" % prefix].iloc[0])
    std = float(base["%s_std" % prefix].iloc[0]) or 1.0
    lo = (float(base["%s_min" % prefix].iloc[0]) - mean) / std
    hi = (float(base["%s_max" % prefix].iloc[0]) - mean) / std
    return max(abs(lo), abs(hi))


def add_sigma_trace(fig, df, channel, base, row, colour):
    """Plot the bucket peak as sigma above the clean baseline.

    Every channel is drawn in the same unit - standard deviations from its
    own clean-only mean - so the panels are directly comparable without a
    second y-axis.
    """
    prefix = CHANNEL_PREFIX[channel]
    mean = float(base["%s_mean" % prefix].iloc[0])
    std = float(base["%s_std" % prefix].iloc[0]) or 1.0
    raw = df[channel + "_max"]
    fig.add_trace(
        go.Scatter(
            x=df["frame_mid"], y=(raw - mean) / std,
            name=CHANNEL_LABELS.get(channel, channel),
            line=dict(color=colour, width=2),
            customdata=np.asarray(raw),
            hovertemplate="%{x}<br>%{y:.2f} sigma"
                          "<br>raw %{customdata:.1f}<extra></extra>",
        ),
        row=row, col=1,
    )


def render_insight() -> None:
    st.subheader("What actually precedes an emergency stop")

    version = q.data_version()
    sig = cached_signature(version)
    esc = cached_escalation(version)
    base = cached_baselines(version)

    # ---- headline, every number computed from the flagged view ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emergency stops", fmt_int(sig["n_estop"]))
    c2.metric("...with vibration spike", "%.1f%%" % sig["vib_pct"])
    c3.metric("...with temperature spike", "%.1f%%" % sig["temp_pct"])
    c4.metric("...with op_state = Fault", "%.1f%%" % sig["fault_pct"])

    st.success(
        "Every one of the %s emergency stops carries a vibration spike, a "
        "temperature spike and op_state=4 simultaneously. Vibration spikes run "
        "at %.3f%% on e-stop rows against %.4f%% everywhere else - a **%.0fx "
        "enrichment**."
        % (fmt_int(sig["n_estop"]), sig["vib_pct"], sig["other_vib_pct"],
           sig["enrichment"])
    )
    st.info(
        "Motor speed and oil pressure are cyclic carriers, not predictors: "
        "their correlation with emergency_stop is %+.4f and %+.4f (vibration "
        "%+.4f, temperature %+.4f). They oscillate smoothly through every "
        "fault and **do not predict or explain it** - neither channel can even "
        "reach the %g-sigma threshold, peaking at %.2f and %.2f sigma."
        % (sig["corr_rpm"], sig["corr_psi"], sig["corr_vib"], sig["corr_temp"],
           an.Z_FLAG, peak_sigma(base, "rpm"), peak_sigma(base, "psi"))
    )

    st.divider()

    # ---- the overlay -----------------------------------------------------
    estop_files = cached_estop_files(version)
    files = cached_source_files(version)
    default = (estop_files.iloc[0]["source_file"]
               if len(estop_files) else files[0])
    picked = st.selectbox("Source file", files, index=files.index(default),
                          key="insight_file")

    channels = SPIKE_PANEL + CARRIER_PANEL
    df = cached_series(version, picked, tuple(channels), q.DEFAULT_BUCKETS)
    estops = cached_estops(version, picked)
    raw_rows = cached_frame_count(version, picked)

    st.caption(
        "%s frames bucketed to %s points in DuckDB (%.0f:1), plotting each "
        "bucket's PEAK so single-frame spikes survive. Both panels use the "
        "same unit - sigma above each channel's clean-only baseline - so they "
        "are directly comparable; note the y-ranges differ."
        % (fmt_int(raw_rows), fmt_int(len(df)), raw_rows / max(len(df), 1))
    )

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=[
            "Fault signature: vibration + cylinder temperature",
            "Cyclic carriers: motor speed + oil pressure (no fault signal)",
        ],
    )
    for channel, colour in zip(SPIKE_PANEL, (SERIES_1, SERIES_2)):
        add_sigma_trace(fig, df, channel, base, 1, colour)
    for channel, colour in zip(CARRIER_PANEL, (SERIES_1, SERIES_2)):
        add_sigma_trace(fig, df, channel, base, 2, colour)

    # The detection threshold, drawn on both panels as the shared reference
    # that makes the scale difference legible.
    for row in (1, 2):
        fig.add_hline(y=an.Z_FLAG, line=dict(color=GRID, width=1, dash="dot"),
                      annotation_text="z = %g spike threshold" % an.Z_FLAG,
                      annotation_position="top left", row=row, col=1)

    mark_estops(fig, estops, 2)
    style_figure(fig, 2, height_per=260)
    fig.update_yaxes(title_text="sigma vs clean baseline")
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ---- escalation table ------------------------------------------------
    st.subheader("Operating-state escalation")
    labels = esc["op_state"].map(lambda s: dc.OP_STATE_LABELS.get(int(s), "?"))
    table = esc.copy()
    table.insert(1, "state", labels)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "op_state": st.column_config.NumberColumn("code", format="%d"),
            "state": st.column_config.TextColumn("state"),
            "rows": st.column_config.NumberColumn("packets", format="%d"),
            "estops": st.column_config.NumberColumn("e-stops", format="%d"),
            "vib_spikes": st.column_config.NumberColumn("vib spikes", format="%d"),
            "vib_avg": st.column_config.NumberColumn("vib avg", format="%.1f"),
            "vib_max": st.column_config.NumberColumn("vib max", format="%.0f"),
            "temp_avg": st.column_config.NumberColumn("temp avg", format="%.1f"),
            "rpm_avg": st.column_config.NumberColumn("rpm avg", format="%.1f"),
            "psi_avg": st.column_config.NumberColumn("psi avg", format="%.2f"),
        },
    )
    st.caption(
        "Vibration escalates %.0f -> %.0f -> %.0f across Run -> Warning -> Fault "
        "while temperature jumps once and then saturates (%.1f -> %.1f -> %.1f) "
        "and motor speed and oil pressure barely move "
        "(%.0f -> %.0f RPM, %.2f -> %.2f PSI). Vibration "
        "magnitude is what separates a warning from a fault."
        % (esc["vib_avg"].iloc[0], esc["vib_avg"].iloc[1], esc["vib_avg"].iloc[2],
           esc["temp_avg"].iloc[0], esc["temp_avg"].iloc[1], esc["temp_avg"].iloc[2],
           esc["rpm_avg"].iloc[0], esc["rpm_avg"].iloc[2],
           esc["psi_avg"].iloc[0], esc["psi_avg"].iloc[2])
    )


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------

def frame_split(n_bytes: int) -> tuple[int, int]:
    """Split a byte count into (whole-frame bytes, leftover bytes)."""
    usable = (n_bytes // dc.FRAME_BYTES) * dc.FRAME_BYTES
    return usable, n_bytes - usable


def import_one_file(data: bytes, filename: str) -> dict:
    """Write one raw file, decode it, and refresh everything downstream.

    Reuses ingest.ingest_file unchanged, so an imported file goes through the
    exact same path as the original corpus - one file in memory at a time.
    """
    con = get_connection()
    before = q.total_packets(con)

    target = os.path.join(ingest.RAW_DIR, filename)
    os.makedirs(ingest.RAW_DIR, exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(data)

    out_path, rows = ingest.ingest_file(target, dbmod.DECODED_DIR)

    # The view globs decoded/, so the new Parquet is picked up on refresh;
    # baselines are recomputed so spike thresholds account for the new data.
    dbmod.refresh_view(con)
    an.build(con)

    # data_version() has changed (new mtime in decoded/), but clear explicitly
    # so nothing cached against the old fingerprint can survive.
    st.cache_data.clear()

    after = q.total_packets(con)
    return {
        "before": before, "after": after, "rows": rows,
        "parquet": os.path.basename(out_path), "source_file": filename,
    }


def render_import() -> None:
    st.subheader("Import a raw telemetry file")
    st.caption(
        "One file at a time. The upload is decoded through the same "
        "ingest.ingest_file path as the original corpus and written to its own "
        "Parquet file, so memory stays bounded by a single file."
    )

    uploaded = st.file_uploader(
        "Raw telemetry (.raw or .bin)", type=["raw", "bin"],
        accept_multiple_files=False, key="import_uploader",
    )

    if uploaded is None:
        st.info(
            "Expected format: %d-byte frames (%d bits), MSB-first, sync 0xAA55, "
            "CRC-16/CCITT-FALSE over bits 0-183."
            % (dc.FRAME_BYTES, dc.FRAME_BITS)
        )
        return

    data = uploaded.getvalue()
    usable, leftover = frame_split(len(data))
    n_frames = usable // dc.FRAME_BYTES

    m1, m2, m3 = st.columns(3)
    m1.metric("Upload size", "%s bytes" % fmt_int(len(data)))
    m2.metric("Whole frames", fmt_int(n_frames))
    m3.metric("Leftover bytes", fmt_int(leftover))

    # ---- size guard ------------------------------------------------------
    if leftover:
        st.warning(
            "Size %s is not a multiple of %d bytes. The trailing %d byte(s) "
            "do not form a complete frame and will be ignored; %s whole frames "
            "(%s bytes) will be decoded."
            % (fmt_int(len(data)), dc.FRAME_BYTES, leftover,
               fmt_int(n_frames), fmt_int(usable))
        )
    if n_frames == 0:
        st.error(
            "This file contains no complete %d-byte frame, so there is nothing "
            "to decode." % dc.FRAME_BYTES
        )
        return

    # ---- duplicate guard -------------------------------------------------
    target = os.path.join(ingest.RAW_DIR, uploaded.name)
    exists = os.path.exists(target)
    overwrite = False
    if exists:
        st.warning(
            "**%s** already exists in RAW-DATA/. Importing it again will "
            "replace both the raw file and its decoded Parquet."
            % uploaded.name
        )
        overwrite = st.checkbox("Yes, overwrite the existing file",
                                key="import_overwrite")

    blocked = exists and not overwrite
    go = st.button("Decode and add to corpus", type="primary",
                   disabled=blocked, key="import_go")
    if blocked:
        st.caption("Tick the overwrite box to enable importing.")

    if not go:
        return

    with st.spinner("Decoding %s..." % uploaded.name):
        result = import_one_file(data, uploaded.name)

    con = get_connection()
    summary = q.file_summary(con, uploaded.name)

    st.success("Imported %s - %s packets decoded into %s."
               % (result["source_file"], fmt_int(result["rows"]),
                  result["parquet"]))

    b1, b2, b3 = st.columns(3)
    b1.metric("Packets before", fmt_int(result["before"]))
    b2.metric("Packets after", fmt_int(result["after"]),
              delta="%+d" % (result["after"] - result["before"]))
    b3.metric("This file", fmt_int(summary["packets"]))

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Sync pass", "%.4f%%" % summary["sync_pct"])
    n2.metric("CRC pass", "%.4f%%" % summary["crc_pct"])
    n3.metric("Emergency stops", fmt_int(summary["estops"]))
    n4.metric("Vibration spikes", fmt_int(summary["vib_spikes"]))

    st.caption(
        "The DuckDB view and the clean-only baselines were refreshed, so this "
        "file now appears in every tab."
    )


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------

def main() -> None:
    st.title("IoT Telemetry Decoder & Analyzer")
    st.caption(
        "200-bit frames, 25 bytes, 14 Hz - MSB-first - "
        "CRC-16/CCITT-FALSE over bits 0-183"
    )

    if not q.has_data():
        st.warning(
            "No decoded data found. Run `python ingest.py` to build "
            "decoded/*.parquet, or add a file from the Import tab."
        )
        st.stop()

    tab1, tab_range, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Range analysis", "Schema editor", "Time series",
         "Insight", "Import"]
    )

    with tab1:
        render_overview()
    with tab_range:
        render_range_analysis()
    with tab2:
        render_schema_editor()
    with tab3:
        render_time_series()
    with tab4:
        render_insight()
    with tab5:
        render_import()


if __name__ == "__main__":
    main()

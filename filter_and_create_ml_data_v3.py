#!/usr/bin/env python3
"""
build_road_event_history_ml_filtered.py

Create ONE unified chronological event-history parquet file from the wide raw road dataset,
with additional ML-oriented filtering applied during PTM extraction.

Final output:
  OUT_DIR/OUT_FINAL_SINGLE

Design:
  - Stream-read the wide parquet row-group by row-group
  - Filter segments to 100 m only
  - Extract filtered PTM measurement events and TP treatment events into temporary bucketed parquet
  - For each bucket:
      * normalize events into one unified event table
      * compute PTM-derived transition fields
      * compute TP interval counts for PTM rows
      * infer resets / lifecycles from PTM rows
      * propagate lifecycle context into unified chronology
  - Consolidate all bucket outputs into ONE final parquet file

Requirements:
  - pyarrow
  - pandas
  - numpy
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# ============================================================
# HARD-CODED CONFIG
# ============================================================
IN_PARQUET = "data/historiadata_ALL.parquet"
OUT_DIR = "data"
OUT_FINAL_SINGLE = "road_event_history_v2.parquet"

# Keys
KEY_COLS = ["ELY", "Tie", "Ajorata", "Kaista", "Aosa", "Aet", "Losa", "Let"]

# Data quality / cleanup
# TP_MIN_VALID_DATE = pd.Timestamp("1950-01-01")
TP_MIN_VALID_DATE = pd.Timestamp("2005-01-01")
TP_PLACEHOLDER_YEAR = 1900
# PTM_MIN_VALID_DATE = pd.Timestamp("1950-01-01")
PTM_MIN_VALID_DATE = pd.Timestamp("2005-01-01")

IRI_MIN, IRI_MAX = 0.0, 20.0
URA_MIN, URA_MAX = 0.0, 80.0
MIN_VALID_TP_YEAR = 2005
MIN_VALID_PTM_YEAR = 2005

# ML filtering thresholds
ML_REQUIRED_SEGMENT_LENGTH = 100.0
ML_MAX_IRI = 10.0
ML_MAX_URA = 40.0

# Reset logic thresholds (mm)
KNOWN_TP_RESET_DROP_MM = -1.0
PHANTOM_RESET_DROP_MM = -3.0

# Wide-slot scan caps
PTM_MAX_SLOT_SCAN = 120
TP_MAX_SLOT_SCAN = 120

# Memory knobs
N_BUCKETS = 512
MAX_ROWS_PER_CHUNK = 250_000

# Compression
PARQUET_COMPRESSION = "zstd"
# ============================================================


# -----------------------------
# Helpers
# -----------------------------
def indexed_cols_from_names(names: List[str], base: str) -> List[str]:
    out: List[Tuple[int, str]] = []
    pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
    for c in names:
        m = pat.match(str(c))
        if m:
            out.append((int(m.group(1)), c))
    return [c for _, c in sorted(out, key=lambda x: x[0])]


def to_dt(s: pd.Series) -> pd.Series:
    return s if np.issubdtype(s.dtype, np.datetime64) else pd.to_datetime(s, errors="coerce")


def to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    s2 = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def clamp_series_to_nan(s: pd.Series, lo: float, hi: float) -> pd.Series:
    s2 = to_num(s)
    m = s2.notna() & ((s2 < lo) | (s2 > hi))
    if m.any():
        s2 = s2.copy()
        s2.loc[m] = np.nan
    return s2


def ensure_keys_exist_in_names(all_cols: List[str], keys: List[str]) -> None:
    missing = [c for c in keys if c not in all_cols]
    if missing:
        raise ValueError(f"Missing required key columns: {missing}")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def segment_hash_bucket(df_keys: pd.DataFrame, n_buckets: int) -> np.ndarray:
    h = pd.util.hash_pandas_object(df_keys, index=False).astype("uint64").to_numpy()
    b = (h % np.uint64(n_buckets)).astype(np.uint16 if n_buckets <= 65535 else np.uint32)
    return b


def write_bucketed_parquet(
    base_dir: Path,
    bucket_ids: np.ndarray,
    df: pd.DataFrame,
    bucket_col: str = "_bucket",
    compression: str = PARQUET_COMPRESSION,
) -> None:
    df = df.copy()
    df[bucket_col] = bucket_ids

    for b, part in df.groupby(bucket_col, sort=False):
        bucket_dir = base_dir / f"bucket={int(b):05d}"
        safe_mkdir(bucket_dir)
        part_file = bucket_dir / f"part-{np.random.randint(0, 1_000_000_000):09d}.parquet"
        table = pa.Table.from_pandas(part.drop(columns=[bucket_col]), preserve_index=False)
        pq.write_table(table, part_file, compression=compression)


def load_bucket_dataset(bucket_dir: Path) -> pd.DataFrame:
    if not bucket_dir.exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(bucket_dir), format="parquet")
    table = dataset.to_table()
    return table.to_pandas(split_blocks=True, self_destruct=True)


def consolidate_dataset_to_single_file(dataset_dir: Path, out_file: Path) -> None:
    dataset = ds.dataset(str(dataset_dir), format="parquet")
    table = dataset.to_table()
    pq.write_table(table, out_file, compression=PARQUET_COMPRESSION)


def count_tp_per_interval(meas_dates_sorted: np.ndarray, tp_dates_sorted: np.ndarray) -> np.ndarray:
    """
    Count TP events in interval (prev_meas_date, meas_date] for each measurement date.
    """
    n = meas_dates_sorted.shape[0]
    if tp_dates_sorted.size == 0:
        return np.zeros(n, dtype=np.int32)

    right = np.searchsorted(tp_dates_sorted, meas_dates_sorted, side="right")
    prev = np.concatenate(([np.datetime64("1900-01-01")], meas_dates_sorted[:-1]))
    left = np.searchsorted(tp_dates_sorted, prev, side="right")
    return (right - left).astype(np.int32)


def build_segment_id(df: pd.DataFrame) -> pd.Series:
    return df[KEY_COLS].astype(str).agg("_".join, axis=1)


def print_quality_counts(events: pd.DataFrame) -> None:
    missing_segment_id = int(events["Segment_ID"].isna().sum()) if "Segment_ID" in events.columns else -1
    invalid_event_type = int((~events["event_type"].isin(["PTM", "TP"])).sum()) if "event_type" in events.columns else -1
    invalid_event_order = int((~events["event_order"].isin([0, 1])).sum()) if "event_order" in events.columns else -1

    ptm_rows = events["event_type"].eq("PTM")
    tp_rows = events["event_type"].eq("TP")

    ptm_rows_missing_ptm_idx = int(events.loc[ptm_rows, "ptm_idx"].isna().sum()) if "ptm_idx" in events.columns else -1
    tp_rows_missing_tp_idx = int(events.loc[tp_rows, "tp_idx"].isna().sum()) if "tp_idx" in events.columns else -1

    ptm_rows_missing_ura = int(events.loc[ptm_rows, "URA"].isna().sum()) if "URA" in events.columns else -1
    ptm_rows_missing_iri = int(events.loc[ptm_rows, "IRI"].isna().sum()) if "IRI" in events.columns else -1

    print(
        "missing Segment_ID", missing_segment_id,
        "invalid event_type", invalid_event_type,
        "invalid event_order", invalid_event_order,
        "PTM rows missing ptm_idx", ptm_rows_missing_ptm_idx,
        "TP rows missing tp_idx", tp_rows_missing_tp_idx,
        "PTM rows missing URA", ptm_rows_missing_ura,
        "PTM rows missing IRI", ptm_rows_missing_iri,
    )


# -----------------------------
# Stage 1: stream wide parquet -> bucketed PTM and TP events
# -----------------------------
def stage1_extract_events_streaming(
    in_parquet: str,
    tmp_tp_dir: Path,
    tmp_ptm_dir: Path,
    n_buckets: int,
) -> Dict[str, List[str]]:
    pf = pq.ParquetFile(in_parquet)
    all_cols = pf.schema.names
    ensure_keys_exist_in_names(all_cols, KEY_COLS)

    ptm_date_cols = indexed_cols_from_names(all_cols, "PTM_pvm")
    iri_cols = indexed_cols_from_names(all_cols, "Iri")
    ura_cols = indexed_cols_from_names(all_cols, "Ura_max")

    tp_date_cols = indexed_cols_from_names(all_cols, "Tp_pvm")
    tp_pinta_cols = indexed_cols_from_names(all_cols, "Tp_pinta")
    tp_tyomen_cols = indexed_cols_from_names(all_cols, "Tp_työmen")
    if not tp_tyomen_cols:
        tp_tyomen_cols = indexed_cols_from_names(all_cols, "Tp_tyomen")

    static_candidates = [
        "Nopeus",
        "KVL",
        "KVL_raskas",
        "KVL_kaista",
        "Toim_lk",
        "Pituus",
    ]
    static_cols = [c for c in (KEY_COLS + static_candidates) if c in all_cols]

    use_cols = list(dict.fromkeys(KEY_COLS + static_cols))

    # PTM slots
    for i in range(1, PTM_MAX_SLOT_SCAN + 1):
        for c in (f"PTM_pvm_{i}", f"Iri_{i}", f"Ura_max_{i}"):
            if c in all_cols:
                use_cols.append(c)

    # TP slots
    for i in range(1, TP_MAX_SLOT_SCAN + 1):
        for c in (f"Tp_pvm_{i}", f"Tp_pinta_{i}", f"Tp_työmen_{i}", f"Tp_tyomen_{i}"):
            if c in all_cols:
                use_cols.append(c)

    safe_mkdir(tmp_tp_dir)
    safe_mkdir(tmp_ptm_dir)

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=use_cols)
        df = table.to_pandas(split_blocks=True, self_destruct=True)

        if len(df) > MAX_ROWS_PER_CHUNK:
            for start in range(0, len(df), MAX_ROWS_PER_CHUNK):
                chunk = df.iloc[start : start + MAX_ROWS_PER_CHUNK].copy()
                _process_wide_chunk_to_events(
                    chunk,
                    tmp_tp_dir=tmp_tp_dir,
                    tmp_ptm_dir=tmp_ptm_dir,
                    n_buckets=n_buckets,
                    static_cols=static_cols,
                )
        else:
            _process_wide_chunk_to_events(
                df.copy(),
                tmp_tp_dir=tmp_tp_dir,
                tmp_ptm_dir=tmp_ptm_dir,
                n_buckets=n_buckets,
                static_cols=static_cols,
            )

        del df

    return {
        "ptm_date_cols": ptm_date_cols,
        "iri_cols": iri_cols,
        "ura_cols": ura_cols,
        "tp_date_cols": tp_date_cols,
        "tp_pinta_cols": tp_pinta_cols,
        "tp_tyomen_cols": tp_tyomen_cols,
        "static_cols": static_cols,
    }


def _process_wide_chunk_to_events(
    df: pd.DataFrame,
    tmp_tp_dir: Path,
    tmp_ptm_dir: Path,
    n_buckets: int,
    static_cols: List[str],
) -> None:
    # -------------------------------------------------
    # Memory-safe early segment-length filter
    # -------------------------------------------------
    if "Pituus" in df.columns:
        pituus_num = to_num(df["Pituus"])
        keep_seg = pd.Series(
            np.isclose(pituus_num.to_numpy(), ML_REQUIRED_SEGMENT_LENGTH),
            index=df.index,
        )
        if not keep_seg.any():
            return
        df = df.loc[keep_seg].copy()
        del pituus_num, keep_seg

    if df.empty:
        return

    keys = df[KEY_COLS].copy()

    # -------------------------
    # Extract PTM events
    # -------------------------
    ptm_parts = []

    for i in range(1, PTM_MAX_SLOT_SCAN + 1):
        dc = f"PTM_pvm_{i}"
        if dc not in df.columns:
            continue

        date_s = to_dt(df[dc])
        m = date_s.notna()
        if m.any():
            valid = ~(date_s.dt.year < MIN_VALID_PTM_YEAR)
            m = m & valid

        if not m.any():
            continue

        ic = f"Iri_{i}"
        uc = f"Ura_max_{i}"

        iri = clamp_series_to_nan(df[ic], IRI_MIN, IRI_MAX) if ic in df.columns else pd.Series(np.nan, index=df.index)
        ura = clamp_series_to_nan(df[uc], URA_MIN, URA_MAX) if uc in df.columns else pd.Series(np.nan, index=df.index)

        # -------------------------------------------------
        # Requested ML filter:
        #   - missing IRI/URA out
        #   - IRI > 10 out
        #   - URA > 40 out
        # -------------------------------------------------
        m = m & iri.notna() & ura.notna() & (iri <= ML_MAX_IRI) & (ura <= ML_MAX_URA)

        if not m.any():
            continue

        tmp = keys.loc[m].copy()
        for sc in static_cols:
            if sc in df.columns and sc not in KEY_COLS:
                tmp[sc] = df.loc[m, sc].to_numpy()

        tmp["event_date"] = date_s.loc[m].to_numpy(dtype="datetime64[ns]")
        tmp["event_type"] = "PTM"
        tmp["event_order"] = np.int8(1)  # TP before PTM on same day
        tmp["ptm_idx"] = np.int16(i)
        tmp["tp_idx"] = pd.Series([pd.NA] * len(tmp), dtype="Int16").to_numpy()

        tmp["IRI"] = iri.loc[m].to_numpy()
        tmp["URA"] = ura.loc[m].to_numpy()

        tmp["Tp_pinta"] = pd.Series([pd.NA] * len(tmp), dtype="object").to_numpy()
        tmp["Tp_tyomen"] = pd.Series([pd.NA] * len(tmp), dtype="object").to_numpy()

        ptm_parts.append(tmp)

    if ptm_parts:
        ptm_long = pd.concat(ptm_parts, ignore_index=True)
        ptm_buckets = segment_hash_bucket(ptm_long[KEY_COLS], n_buckets)
        write_bucketed_parquet(tmp_ptm_dir, ptm_buckets, ptm_long)

    # -------------------------
    # Clean TP date columns once
    # -------------------------
    tp_date_cache: Dict[int, pd.Series] = {}
    for i in range(1, TP_MAX_SLOT_SCAN + 1):
        dc = f"Tp_pvm_{i}"
        if dc not in df.columns:
            continue
        s = to_dt(df[dc])
        m = s.notna() & (s.dt.year == TP_PLACEHOLDER_YEAR)
        if m.any():
            s = s.copy()
            s.loc[m] = pd.NaT
        m2 = s.notna() & (s < TP_MIN_VALID_DATE)
        if m2.any():
            s = s.copy()
            s.loc[m2] = pd.NaT
        tp_date_cache[i] = s

    # -------------------------
    # Extract TP events
    # -------------------------
    tp_parts = []

    for i in range(1, TP_MAX_SLOT_SCAN + 1):
        dc = f"Tp_pvm_{i}"
        if dc not in df.columns or i not in tp_date_cache:
            continue

        date_s = tp_date_cache[i]
        m = date_s.notna() & (date_s.dt.year >= MIN_VALID_TP_YEAR)
        if not m.any():
            continue

        pinta_col = f"Tp_pinta_{i}"
        tyomen_col = f"Tp_työmen_{i}"
        if tyomen_col not in df.columns:
            tyomen_col = f"Tp_tyomen_{i}"

        tmp = keys.loc[m].copy()
        for sc in static_cols:
            if sc in df.columns and sc not in KEY_COLS:
                tmp[sc] = df.loc[m, sc].to_numpy()

        tmp["event_date"] = date_s.loc[m].to_numpy(dtype="datetime64[ns]")
        tmp["event_type"] = "TP"
        tmp["event_order"] = np.int8(0)  # TP before PTM on same day
        tmp["ptm_idx"] = pd.Series([pd.NA] * len(tmp), dtype="Int16").to_numpy()
        tmp["tp_idx"] = np.int16(i)

        tmp["IRI"] = np.nan
        tmp["URA"] = np.nan

        if pinta_col in df.columns:
            tmp["Tp_pinta"] = df.loc[m, pinta_col].astype("object").to_numpy()
        else:
            tmp["Tp_pinta"] = pd.Series([pd.NA] * len(tmp), dtype="object").to_numpy()

        if tyomen_col in df.columns:
            tmp["Tp_tyomen"] = df.loc[m, tyomen_col].astype("object").to_numpy()
        else:
            tmp["Tp_tyomen"] = pd.Series([pd.NA] * len(tmp), dtype="object").to_numpy()

        tp_parts.append(tmp)

    if tp_parts:
        tp_long = pd.concat(tp_parts, ignore_index=True)
        tp_buckets = segment_hash_bucket(tp_long[KEY_COLS], n_buckets)
        write_bucketed_parquet(tmp_tp_dir, tp_buckets, tp_long)


# -----------------------------
# Stage 2: build unified event history bucket by bucket
# -----------------------------
def build_event_history_for_bucket(ptm_df: pd.DataFrame, tp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one unified chronological event table for a single bucket.

    Strategy:
      1. clean + deduplicate PTM and TP separately
      2. compute PTM-derived transition / lifecycle fields from PTM rows
      3. compute TP context from surrounding PTM rows
      4. concatenate back into one chronological event history
    """
    if ptm_df.empty and tp_df.empty:
        return pd.DataFrame()

    # -------------------------
    # Clean PTM rows
    # -------------------------
    if not ptm_df.empty:
        ptm = ptm_df.copy()
        ptm["event_date"] = pd.to_datetime(ptm["event_date"], errors="coerce")
        ptm = ptm.dropna(subset=["event_date"]).copy()
        ptm = ptm.drop_duplicates(subset=KEY_COLS + ["event_date", "event_type", "ptm_idx"])
        ptm["Segment_ID"] = build_segment_id(ptm)
        ptm = ptm.sort_values(KEY_COLS + ["event_date", "ptm_idx"], kind="mergesort").reset_index(drop=True)
    else:
        ptm = pd.DataFrame()

    # -------------------------
    # Clean TP rows
    # -------------------------
    if not tp_df.empty:
        tp = tp_df.copy()
        tp["event_date"] = pd.to_datetime(tp["event_date"], errors="coerce")
        tp = tp.dropna(subset=["event_date"]).copy()

        tp_subset = KEY_COLS + ["event_date", "event_type", "tp_idx"]
        if "Tp_pinta" in tp.columns:
            tp_subset.append("Tp_pinta")
        if "Tp_tyomen" in tp.columns:
            tp_subset.append("Tp_tyomen")

        tp = tp.drop_duplicates(subset=tp_subset)
        tp["Segment_ID"] = build_segment_id(tp)
        tp = tp.sort_values(KEY_COLS + ["event_date", "tp_idx"], kind="mergesort").reset_index(drop=True)
    else:
        tp = pd.DataFrame()

    # -------------------------
    # Enrich PTM rows
    # -------------------------
    if not ptm.empty:
        ptm = _compute_ptm_context(ptm, tp)
        ptm = _compute_lifecycles_from_ptm(ptm)
        ptm["days_since_prev_meas"] = ptm["Delta_t_days"]
        ptm["days_until_next_meas"] = (
            pd.to_datetime(ptm["next_meas_date"]) - pd.to_datetime(ptm["event_date"])
        ).dt.days
    else:
        ptm = pd.DataFrame()

    # -------------------------
    # Enrich TP rows from surrounding PTM rows
    # -------------------------
    if not tp.empty:
        tp = _compute_tp_context(tp, ptm)
    else:
        tp = pd.DataFrame()

    # Optional: keep only segments that still have PTM rows after filtering
    # This avoids segments containing only TP rows with no valid measurements left.
    if not ptm.empty and not tp.empty:
        valid_segments = set(ptm["Segment_ID"].dropna().unique())
        tp = tp[tp["Segment_ID"].isin(valid_segments)].copy()
    elif ptm.empty:
        return pd.DataFrame()

    # -------------------------
    # Recombine into one unified chronology
    # -------------------------
    if ptm.empty:
        events = tp.copy()
    elif tp.empty:
        events = ptm.copy()
    else:
        events = pd.concat([ptm, tp], ignore_index=True, sort=False)

    if events.empty:
        return events

    events = events.sort_values(
        ["Segment_ID", "event_date", "event_order", "event_type", "ptm_idx", "tp_idx"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    events["Event_Idx"] = events.groupby("Segment_ID").cumcount() + 1
    events["year"] = pd.to_datetime(events["event_date"]).dt.year.astype("Int64")

    preferred = [
        "Segment_ID",
        "Lifecycle_ID",
        "event_date",
        "year",
        "event_type",
        "event_order",
        "Event_Idx",
        "Measurement_Idx",
        "ptm_idx",
        "tp_idx",
        "IRI",
        "URA",
        "prev_IRI",
        "prev_URA",
        "delta_IRI",
        "delta_URA",
        "prev_meas_date",
        "next_meas_date",
        "Delta_t_days",
        "Delta_t_years",
        "days_since_prev_meas",
        "days_until_next_meas",
        "Pavement_Age_years",
        "Initial_URA",
        "tp_count_interval",
        "has_TP_interval",
        "is_minor_treatment",
        "Minor_TP_Count",
        "is_major_reset",
        "is_phantom_reset",
        "cycle_num",
        "Tp_pinta",
        "Tp_tyomen",
        "KVL",
        "KVL_raskas",
        "KVL_kaista",
        "Nopeus",
        "Toim_lk",
        "Pituus",
        *KEY_COLS,
    ]

    final_cols = [c for c in preferred if c in events.columns] + [c for c in events.columns if c not in preferred]
    return events[final_cols]


def _compute_ptm_context(ptm: pd.DataFrame, tp_df: pd.DataFrame) -> pd.DataFrame:
    ptm = ptm.sort_values(KEY_COLS + ["event_date", "ptm_idx"], kind="mergesort").reset_index(drop=True)

    ptm["prev_meas_date"] = ptm.groupby(KEY_COLS)["event_date"].shift(1)
    ptm["next_meas_date"] = ptm.groupby(KEY_COLS)["event_date"].shift(-1)

    ptm["prev_URA"] = ptm.groupby(KEY_COLS)["URA"].shift(1)
    ptm["prev_IRI"] = ptm.groupby(KEY_COLS)["IRI"].shift(1)

    ptm["delta_URA"] = ptm["URA"] - ptm["prev_URA"]
    ptm["delta_IRI"] = ptm["IRI"] - ptm["prev_IRI"]

    ptm["Delta_t_days"] = (
        pd.to_datetime(ptm["event_date"]) - pd.to_datetime(ptm["prev_meas_date"])
    ).dt.days
    ptm["Delta_t_years"] = ptm["Delta_t_days"] / 365.25

    first_obs = ptm["prev_meas_date"].isna()
    ptm.loc[first_obs, ["Delta_t_days", "Delta_t_years", "delta_URA", "delta_IRI"]] = np.nan

    # Count TP events in each measurement interval
    tp_map: Dict[Tuple, np.ndarray] = {}

    if not tp_df.empty:
        tp_work = tp_df.dropna(subset=["event_date"]).copy()
        tp_work["event_date"] = pd.to_datetime(tp_work["event_date"], errors="coerce")
        tp_work = tp_work.dropna(subset=["event_date"])

        segs = ptm[KEY_COLS].drop_duplicates()
        tp_work = tp_work.merge(segs, on=KEY_COLS, how="inner")

        for k, g in tp_work.groupby(KEY_COLS, sort=False):
            tp_map[k] = np.sort(g["event_date"].to_numpy(dtype="datetime64[ns]"))

    counts = np.zeros(len(ptm), dtype=np.int32)

    for k, idx in ptm.groupby(KEY_COLS, sort=False).indices.items():
        meas_arr = ptm.loc[idx, "event_date"].to_numpy(dtype="datetime64[ns]")
        tp_arr = tp_map.get(k, np.array([], dtype="datetime64[ns]"))
        counts[idx] = count_tp_per_interval(meas_arr, tp_arr)

    ptm["tp_count_interval"] = counts
    ptm["has_TP_interval"] = ptm["tp_count_interval"] > 0

    return ptm


def _compute_tp_context(tp: pd.DataFrame, ptm: pd.DataFrame) -> pd.DataFrame:
    """
    Add surrounding PTM context to TP rows.
    """
    tp = tp.copy()

    init_cols = {
        "Lifecycle_ID": pd.Series(pd.NA, index=tp.index, dtype="object"),
        "Measurement_Idx": pd.Series(pd.NA, index=tp.index, dtype="Int64"),
        "prev_IRI": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "prev_URA": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "delta_IRI": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "delta_URA": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "prev_meas_date": pd.Series(pd.NaT, index=tp.index, dtype="datetime64[ns]"),
        "next_meas_date": pd.Series(pd.NaT, index=tp.index, dtype="datetime64[ns]"),
        "Delta_t_days": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "Delta_t_years": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "days_since_prev_meas": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "days_until_next_meas": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "Pavement_Age_years": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "Initial_URA": pd.Series(np.nan, index=tp.index, dtype="float64"),
        "tp_count_interval": pd.Series(pd.NA, index=tp.index, dtype="Int64"),
        "has_TP_interval": pd.Series(pd.NA, index=tp.index, dtype="boolean"),
        "is_minor_treatment": pd.Series(pd.NA, index=tp.index, dtype="boolean"),
        "Minor_TP_Count": pd.Series(pd.NA, index=tp.index, dtype="Int64"),
        "is_major_reset": pd.Series(pd.NA, index=tp.index, dtype="boolean"),
        "is_phantom_reset": pd.Series(pd.NA, index=tp.index, dtype="boolean"),
        "cycle_num": pd.Series(pd.NA, index=tp.index, dtype="Int64"),
    }
    for c, s in init_cols.items():
        tp[c] = s

    if ptm.empty:
        return tp

    ptm = ptm.sort_values(["Segment_ID", "event_date", "ptm_idx"], kind="mergesort").reset_index(drop=True)
    tp = tp.sort_values(["Segment_ID", "event_date", "tp_idx"], kind="mergesort").reset_index(drop=True)

    ptm_lookup: Dict[str, pd.DataFrame] = {}
    for seg, g in ptm.groupby("Segment_ID", sort=False):
        ptm_lookup[seg] = g.reset_index(drop=True)

    out_parts = []

    for seg, g in tp.groupby("Segment_ID", sort=False):
        g = g.copy()
        ref = ptm_lookup.get(seg)

        if ref is None or ref.empty:
            out_parts.append(g)
            continue

        meas_dates = ref["event_date"].to_numpy(dtype="datetime64[ns]")
        ev_dates = g["event_date"].to_numpy(dtype="datetime64[ns]")

        prev_idx = np.searchsorted(meas_dates, ev_dates, side="right") - 1
        next_idx = np.searchsorted(meas_dates, ev_dates, side="left")

        prev_meas = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")
        next_meas = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")

        valid_prev = prev_idx >= 0
        if valid_prev.any():
            prev_meas[valid_prev] = meas_dates[prev_idx[valid_prev]]

        valid_next = next_idx < len(meas_dates)
        if valid_next.any():
            next_meas[valid_next] = meas_dates[next_idx[valid_next]]

        g["prev_meas_date"] = pd.to_datetime(prev_meas)
        g["next_meas_date"] = pd.to_datetime(next_meas)

        g["days_since_prev_meas"] = (
            pd.to_datetime(g["event_date"]) - pd.to_datetime(g["prev_meas_date"])
        ).dt.days

        g["days_until_next_meas"] = (
            pd.to_datetime(g["next_meas_date"]) - pd.to_datetime(g["event_date"])
        ).dt.days

        lifecycle_vals = []
        cycle_vals = []
        initial_ura_vals = []
        pavement_age_vals = []

        for i, p in enumerate(prev_idx):
            if p >= 0:
                lifecycle_vals.append(ref.iloc[p]["Lifecycle_ID"])
                cycle_vals.append(ref.iloc[p]["cycle_num"])
                initial_ura_vals.append(ref.iloc[p]["Initial_URA"])

                cycle_start_date = ref[ref["Lifecycle_ID"] == ref.iloc[p]["Lifecycle_ID"]]["event_date"].min()
                if pd.notna(cycle_start_date):
                    pavement_age_vals.append((g.iloc[i]["event_date"] - cycle_start_date).days / 365.25)
                else:
                    pavement_age_vals.append(np.nan)
            else:
                lifecycle_vals.append(pd.NA)
                cycle_vals.append(pd.NA)
                initial_ura_vals.append(np.nan)
                pavement_age_vals.append(np.nan)

        g["Lifecycle_ID"] = lifecycle_vals
        g["cycle_num"] = pd.Series(cycle_vals, index=g.index, dtype="Int64")
        g["Initial_URA"] = initial_ura_vals
        g["Pavement_Age_years"] = pavement_age_vals

        out_parts.append(g)

    tp_out = pd.concat(out_parts, ignore_index=True, sort=False)
    return tp_out


def _compute_lifecycles_from_ptm(ptm: pd.DataFrame) -> pd.DataFrame:
    ptm = ptm.copy()

    rule_known = ptm["has_TP_interval"] & (ptm["delta_URA"] <= KNOWN_TP_RESET_DROP_MM)
    rule_phantom = (~ptm["has_TP_interval"]) & (ptm["delta_URA"] <= PHANTOM_RESET_DROP_MM)

    ptm["is_major_reset"] = (rule_known | rule_phantom).fillna(False)
    ptm["is_phantom_reset"] = rule_phantom.fillna(False)
    ptm["is_minor_treatment"] = (ptm["has_TP_interval"] & (~ptm["is_major_reset"])).fillna(False)

    ptm["cycle_num"] = ptm.groupby(KEY_COLS)["is_major_reset"].cumsum().astype("Int64")
    ptm["Lifecycle_ID"] = ptm["Segment_ID"] + "_C" + ptm["cycle_num"].astype(str)

    cycle_start = ptm.groupby("Lifecycle_ID")["event_date"].transform("min")
    ptm["Pavement_Age_years"] = (
        (pd.to_datetime(ptm["event_date"]) - pd.to_datetime(cycle_start)).dt.days / 365.25
    ).clip(lower=0)

    ptm["Measurement_Idx"] = ptm.groupby("Lifecycle_ID").cumcount() + 1
    ptm["Measurement_Idx"] = ptm["Measurement_Idx"].astype("Int64")

    ptm["Initial_URA"] = ptm.groupby("Lifecycle_ID")["URA"].transform("first")
    ptm["Minor_TP_Count"] = ptm.groupby("Lifecycle_ID")["is_minor_treatment"].cumsum().astype("Int64")

    return ptm


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    out_dir = Path(OUT_DIR)
    safe_mkdir(out_dir)

    tmp_dir = out_dir / "_tmp_event_history"
    tmp_tp_dir = tmp_dir / "tp_events"
    tmp_ptm_dir = tmp_dir / "ptm_events"
    bucket_out_dir = tmp_dir / "event_history_buckets"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    safe_mkdir(tmp_tp_dir)
    safe_mkdir(tmp_ptm_dir)
    safe_mkdir(bucket_out_dir)

    print(f"Stage 1: streaming extract PTM/TP events from {IN_PARQUET}")
    print(f"  ML segment filter: Pituus == {ML_REQUIRED_SEGMENT_LENGTH}")
    print(f"  ML PTM filters: IRI notna and <= {ML_MAX_IRI}, URA notna and <= {ML_MAX_URA}")

    info = stage1_extract_events_streaming(
        in_parquet=IN_PARQUET,
        tmp_tp_dir=tmp_tp_dir,
        tmp_ptm_dir=tmp_ptm_dir,
        n_buckets=N_BUCKETS,
    )
    print("  Done stage 1.")
    print(f"  PTM date cols found: {len(info['ptm_date_cols'])}")
    print(f"  TP date cols found:  {len(info['tp_date_cols'])}")
    print(f"  Temp PTM dataset: {tmp_ptm_dir}")
    print(f"  Temp TP dataset:  {tmp_tp_dir}")

    print("Stage 2: build unified event-history bucket-by-bucket")
    total_rows = 0
    total_segments = 0

    for b in range(N_BUCKETS):
        ptm_bucket_dir = tmp_ptm_dir / f"bucket={b:05d}"
        tp_bucket_dir = tmp_tp_dir / f"bucket={b:05d}"

        if not ptm_bucket_dir.exists() and not tp_bucket_dir.exists():
            continue

        ptm_df = load_bucket_dataset(ptm_bucket_dir) if ptm_bucket_dir.exists() else pd.DataFrame()
        tp_df = load_bucket_dataset(tp_bucket_dir) if tp_bucket_dir.exists() else pd.DataFrame()

        panel_b = build_event_history_for_bucket(ptm_df, tp_df)
        if panel_b.empty:
            continue

        bucket_out = bucket_out_dir / f"bucket={b:05d}"
        safe_mkdir(bucket_out)
        pq.write_table(
            pa.Table.from_pandas(panel_b, preserve_index=False),
            bucket_out / "part-00000.parquet",
            compression=PARQUET_COMPRESSION,
        )

        total_rows += len(panel_b)
        total_segments += panel_b["Segment_ID"].nunique()

        if b % 50 == 0:
            print(f"  bucket {b:05d}: rows={len(panel_b):,}  segments={panel_b['Segment_ID'].nunique():,}")

    out_file = out_dir / OUT_FINAL_SINGLE
    print(f"Stage 3: consolidate unified event history -> single parquet: {out_file}")
    consolidate_dataset_to_single_file(bucket_out_dir, out_file)

    print(f"Saved final unified event-history parquet: {out_file}")
    print(f"  approx rows={total_rows:,}  approx segment-count-over-buckets={total_segments:,}")

    print("Stage 4: quick final quality check")
    final_df = pd.read_parquet(out_file)
    print_quality_counts(final_df)

    print("Done.")


if __name__ == "__main__":
    main()
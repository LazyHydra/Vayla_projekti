#!/usr/bin/env python3
"""
build_model_dataset_v2.py

Create a modeling-ready PTM dataset from the canonical road event-history parquet.

Changes from v1:
    - Output file renamed to road_model_dataset_v2.parquet
    - Added intervention type classification based on Väylävirasto guidance:
        * tp_surface_type: simplified pavement surface type (AB / PAB / SMA / other / none)
        * tp_material_type: simplified material type (new / recycled / other / none)
      These are derived from Tp_pinta and Tp_tyomen in TP events that occurred in
      the interval ending at each PTM row (i.e. the same interval used for tp_count_interval).
      Values reflect the most recent TP event in the interval, or 'none' if no TP occurred.
    - Tp_pinta and Tp_tyomen are read from the event history for this purpose.

Classification scheme (from Väylävirasto document):
    Surface types:
        AB  (Asfalttibetoni):         ab, abk, abs, ea
        PAB (Pehmeä asfalttibetoni):  pab-v, pab-b, pab-o
        SMA (Kivimastiksi):           sma
        other: any other non-null value
        none: no TP in interval

    Material types:
        new:      lta, mp, mpkj, art, mpk
        recycled: rem, urem, remo, rem+
        other: any other non-null value
        none: no TP in interval

    Väylävirasto note: Various surface-material combinations exist in
    the data (e.g. AB-LTA, SMA-REM, PAB-V-MP). Not all combinations are present.

Requirements:
    - pandas
    - numpy
    - pyarrow (for parquet I/O via pandas)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


# ============================================================
# HARD-CODED CONFIG
# ============================================================
IN_EVENT_HISTORY = Path("data/road_event_history_v2.parquet")
OUT_DIR = Path("data")
OUT_FILE = Path("road_model_dataset_v2.parquet")

DROP_ROWS_WITHOUT_TARGET = True

STATIC_FEATURES = [
    "KVL",
    "KVL_raskas",
    "KVL_kaista",
    "Nopeus",
    "Toim_lk",
    "Pituus",
]

CURRENT_STATE_FEATURES = [
    "IRI",
    "URA",
    "Pavement_Age_years",
    "Initial_URA",
    "Measurement_Idx",
    "Minor_TP_Count",
    "tp_count_interval",
    "has_TP_interval",
]

LAG_FEATURES = [
    "prev_IRI",
    "prev_URA",
    "Delta_t_days",
    "Delta_t_years",
]

IDENTIFIER_COLUMNS = [
    "Segment_ID",
    "Lifecycle_ID",
    "event_date",
    "year",
    "cycle_num",
]

OPTIONAL_CONTEXT_COLUMNS = [
    "is_major_reset",
    "is_phantom_reset",
    "is_minor_treatment",
    "ELY",
    "Tie",
    "Ajorata",
    "Kaista",
    "Aosa",
    "Aet",
    "Losa",
    "Let",
]

# ============================================================
# INTERVENTION CLASSIFICATION MAPS
# ============================================================
# Surface type: maps lowercase Tp_pinta values to simplified category
SURFACE_TYPE_MAP = {
    "ab":    "AB",
    "abk":   "AB",
    "abs":   "AB",
    "ea":    "AB",
    "pab-v": "PAB",
    "pab-b": "PAB",
    "pab-o": "PAB",
    "sma":   "SMA",
}

# Material type: maps lowercase Tp_tyomen values to simplified category
MATERIAL_TYPE_MAP = {
    "lta":  "new",
    "mp":   "new",
    "mpkj": "new",
    "art":  "new",
    "mpk":  "new",
    "rem":  "recycled",
    "urem": "recycled",
    "remo": "recycled",
    "rem+": "recycled",
}
# ============================================================

NUMERIC_MODEL_COLUMNS = [
    "KVL", "KVL_raskas", "KVL_kaista", "Nopeus", "Pituus",
    "IRI", "URA", "Pavement_Age_years", "Initial_URA",
    "Measurement_Idx", "Minor_TP_Count", "tp_count_interval",
    "prev_IRI", "prev_URA", "Delta_t_days", "Delta_t_years",
    "cycle_num", "year",
]

BOOLEAN_MODEL_COLUMNS = [
    "has_TP_interval",
    "is_major_reset",
    "is_phantom_reset",
    "is_minor_treatment",
]

REQUIRED_SOURCE_COLUMNS = [
    "Segment_ID",
    "Lifecycle_ID",
    "event_date",
    "Measurement_Idx",
    "IRI",
    "URA",
]

KEY_COLS = ["ELY", "Tie", "Ajorata", "Kaista", "Aosa", "Aet", "Losa", "Let"]


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Input parquet not found: {path}\n"
            "Run the canonical event-history build first."
        )


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source parquet: {missing}")


def available_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def coerce_boolean_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    return df


def classify_surface_type(val) -> str:
    """Map a raw Tp_pinta value to a simplified surface type category."""
    if pd.isna(val) or str(val).strip() == "":
        return "none"
    return SURFACE_TYPE_MAP.get(str(val).strip().lower(), "other")


def classify_material_type(val) -> str:
    """Map a raw Tp_tyomen value to a simplified material type category."""
    if pd.isna(val) or str(val).strip() == "":
        return "none"
    return MATERIAL_TYPE_MAP.get(str(val).strip().lower(), "other")


def read_all_events(in_path: Path) -> pd.DataFrame:
    """Read both PTM and TP rows — needed to derive intervention features."""
    ptm_cols = (
        ["event_type", "event_date", "prev_meas_date"]
        + REQUIRED_SOURCE_COLUMNS
        + IDENTIFIER_COLUMNS
        + STATIC_FEATURES
        + CURRENT_STATE_FEATURES
        + LAG_FEATURES
        + OPTIONAL_CONTEXT_COLUMNS
    )

    tp_cols = (
        ["event_type", "event_date", "Tp_pinta", "Tp_tyomen"]
        + KEY_COLS
        + ["Segment_ID"]
    )

    all_needed = list(dict.fromkeys(ptm_cols + tp_cols))

    events = pd.read_parquet(in_path, columns=[c for c in all_needed])
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    return events


def build_intervention_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each PTM row derive tp_surface_type and tp_material_type from
    the most recent TP event that occurred in the interval
    (prev_meas_date, event_date] — i.e. the same window used for tp_count_interval.

    Returns the PTM-only dataframe with two new columns added.
    """
    ptm = events[events["event_type"] == "PTM"].copy()
    tp  = events[events["event_type"] == "TP"].copy()

    if tp.empty or "Tp_pinta" not in tp.columns:
        ptm["tp_surface_type"] = "none"
        ptm["tp_material_type"] = "none"
        return ptm

    tp = tp.dropna(subset=["event_date"]).copy()
    tp["tp_surface_type_raw"]  = tp["Tp_pinta"].apply(classify_surface_type)
    tp["tp_material_type_raw"] = tp["Tp_tyomen"].apply(classify_material_type) \
        if "Tp_tyomen" in tp.columns else "none"

    # Index TP events by segment for fast lookup
    tp_by_seg = {}
    for seg, grp in tp.groupby("Segment_ID", sort=False):
        tp_by_seg[seg] = grp.sort_values("event_date").reset_index(drop=True)

    surface_types  = []
    material_types = []

    ptm = ptm.sort_values(["Segment_ID", "event_date"]).reset_index(drop=True)

    for _, row in ptm.iterrows():
        seg = row["Segment_ID"]
        meas_date = row["event_date"]
        prev_date = row.get("prev_meas_date", pd.NaT)

        seg_tp = tp_by_seg.get(seg)
        if seg_tp is None or seg_tp.empty:
            surface_types.append("none")
            material_types.append("none")
            continue

        # TP events in (prev_meas_date, meas_date]
        mask = seg_tp["event_date"] <= meas_date
        if pd.notna(prev_date):
            mask = mask & (seg_tp["event_date"] > prev_date)

        interval_tp = seg_tp[mask]
        if interval_tp.empty:
            surface_types.append("none")
            material_types.append("none")
        else:
            # Use the most recent TP in the interval
            latest = interval_tp.iloc[-1]
            surface_types.append(latest["tp_surface_type_raw"])
            material_types.append(latest["tp_material_type_raw"])

    ptm["tp_surface_type"]  = surface_types
    ptm["tp_material_type"] = material_types

    return ptm


def build_model_table(ptm: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(ptm, REQUIRED_SOURCE_COLUMNS)

    if ptm.empty:
        raise ValueError("No PTM rows found in the source parquet.")

    ptm["event_date"] = pd.to_datetime(ptm["event_date"], errors="coerce")
    ptm = ptm.dropna(subset=["event_date", "Segment_ID", "Lifecycle_ID"]).copy()

    ptm = ptm.sort_values(
        ["Segment_ID", "Lifecycle_ID", "event_date", "Measurement_Idx"],
        kind="mergesort",
    ).reset_index(drop=True)

    ptm = coerce_numeric_columns(ptm, NUMERIC_MODEL_COLUMNS)
    ptm = coerce_boolean_columns(ptm, BOOLEAN_MODEL_COLUMNS)

    by_lifecycle = ptm.groupby("Lifecycle_ID", sort=False)
    ptm["target_next_URA"]              = by_lifecycle["URA"].shift(-1)
    ptm["target_next_IRI"]              = by_lifecycle["IRI"].shift(-1)
    ptm["target_next_event_date"]       = by_lifecycle["event_date"].shift(-1)
    ptm["target_next_measurement_idx"]  = by_lifecycle["Measurement_Idx"].shift(-1)
    ptm["target_horizon_days"] = (
        pd.to_datetime(ptm["target_next_event_date"]) - pd.to_datetime(ptm["event_date"])
    ).dt.days
    ptm["target_horizon_years"] = ptm["target_horizon_days"] / 365.25

    ptm["baseline_persist_URA"] = ptm["URA"]
    ptm["baseline_persist_IRI"] = ptm["IRI"]

    lifecycle_size = by_lifecycle["Measurement_Idx"].transform("max")
    ptm["measurements_remaining_in_lifecycle"]  = lifecycle_size - ptm["Measurement_Idx"]
    ptm["is_first_measurement_in_lifecycle"]    = ptm["Measurement_Idx"].eq(1)
    ptm["has_previous_measurement_in_lifecycle"] = ptm["Measurement_Idx"].gt(1)
    ptm["has_next_measurement_in_lifecycle"]    = ptm["target_next_URA"].notna()

    intervention_cols = []
    if "tp_surface_type" in ptm.columns:
        intervention_cols.append("tp_surface_type")
    if "tp_material_type" in ptm.columns:
        intervention_cols.append("tp_material_type")

    preferred_cols = (
        IDENTIFIER_COLUMNS
        + available_columns(ptm, STATIC_FEATURES)
        + available_columns(ptm, CURRENT_STATE_FEATURES)
        + available_columns(ptm, LAG_FEATURES)
        + intervention_cols
        + [
            "measurements_remaining_in_lifecycle",
            "is_first_measurement_in_lifecycle",
            "has_previous_measurement_in_lifecycle",
            "has_next_measurement_in_lifecycle",
            "target_next_URA",
            "target_next_IRI",
            "target_next_event_date",
            "target_next_measurement_idx",
            "target_horizon_days",
            "target_horizon_years",
            "baseline_persist_URA",
            "baseline_persist_IRI",
        ]
        + available_columns(ptm, OPTIONAL_CONTEXT_COLUMNS)
    )

    model_df = ptm.loc[:, [c for c in preferred_cols if c in ptm.columns]].copy()

    if DROP_ROWS_WITHOUT_TARGET:
        model_df = model_df.dropna(subset=["target_next_URA", "target_next_IRI"]).copy()

    model_df = model_df.sort_values(
        ["Segment_ID", "Lifecycle_ID", "event_date", "target_next_event_date"],
        kind="mergesort",
    ).reset_index(drop=True)

    return model_df


def print_summary(df: pd.DataFrame) -> None:
    print("Model dataset summary")
    print(f"  rows: {len(df):,}")
    print(f"  segments: {df['Segment_ID'].nunique():,}")
    print(f"  lifecycles: {df['Lifecycle_ID'].nunique():,}")

    if not df.empty:
        print(f"  event_date min: {df['event_date'].min()}")
        print(f"  event_date max: {df['event_date'].max()}")
        print(f"  target horizon mean days: {df['target_horizon_days'].mean():.2f}")
        print(f"  target horizon median days: {df['target_horizon_days'].median():.2f}")
        print(f"  missing target_next_URA: {int(df['target_next_URA'].isna().sum())}")
        print(f"  missing target_next_IRI: {int(df['target_next_IRI'].isna().sum())}")

        if "tp_surface_type" in df.columns:
            print("\n  tp_surface_type distribution:")
            print(df["tp_surface_type"].value_counts(dropna=False).to_string())

        if "tp_material_type" in df.columns:
            print("\n  tp_material_type distribution:")
            print(df["tp_material_type"].value_counts(dropna=False).to_string())


def main() -> None:
    ensure_exists(IN_EVENT_HISTORY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading event history: {IN_EVENT_HISTORY}")
    events = read_all_events(IN_EVENT_HISTORY)

    print("Deriving intervention type features from TP events...")
    ptm_rows = build_intervention_features(events)

    print("Building PTM modeling dataset...")
    model_df = build_model_table(ptm_rows)

    out_path = OUT_DIR / OUT_FILE
    model_df.to_parquet(out_path, index=False)

    print(f"Saved model dataset: {out_path}")
    print_summary(model_df)


if __name__ == "__main__":
    main()

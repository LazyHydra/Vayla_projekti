#!/usr/bin/env python3
"""
build_model_dataset.py

Create a modeling-ready PTM dataset from the canonical road event-history parquet.

Why a separate script?
    The event-history parquet is the canonical chronological source. This script
    derives one specific supervised-learning table from it without changing the
    underlying canonical dataset.

Initial modeling design:
    - One row = one PTM measurement event
    - Segment and lifecycle structure is preserved
    - Context comes only from the same segment / lifecycle up to the current row
    - Targets are the next PTM measurement within the same lifecycle

This first version is intentionally simple and suitable for baseline models such as:
    - naive persistence
    - linear regression
    - ridge / lasso

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
OUT_FILE = Path("road_model_dataset_v1.parquet")

# Keep rows only when next target exists inside the same lifecycle.
DROP_ROWS_WITHOUT_TARGET = True

# For the first baseline dataset we keep a compact, interpretable feature set.
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

NUMERIC_MODEL_COLUMNS = [
    "KVL",
    "KVL_raskas",
    "KVL_kaista",
    "Nopeus",
    "Pituus",
    "IRI",
    "URA",
    "Pavement_Age_years",
    "Initial_URA",
    "Measurement_Idx",
    "Minor_TP_Count",
    "tp_count_interval",
    "prev_IRI",
    "prev_URA",
    "Delta_t_days",
    "Delta_t_years",
    "cycle_num",
    "year",
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


def source_columns_to_read() -> List[str]:
    # Only load the columns needed for the first modeling table.
    # This avoids materializing the full event-history parquet in memory.
    cols = (
        ["event_type"]
        + REQUIRED_SOURCE_COLUMNS
        + IDENTIFIER_COLUMNS
        + STATIC_FEATURES
        + CURRENT_STATE_FEATURES
        + LAG_FEATURES
        + OPTIONAL_CONTEXT_COLUMNS
    )
    return list(dict.fromkeys(cols))


def read_ptm_source_rows(in_path: Path) -> pd.DataFrame:
    cols = source_columns_to_read()

    try:
        return pd.read_parquet(
            in_path,
            columns=cols,
            filters=[("event_type", "==", "PTM")],
        )
    except TypeError:
        # Some parquet backends ignore or do not support filters in pandas.
        # Fall back to loading only the required columns and filter afterwards.
        events = pd.read_parquet(in_path, columns=cols)
        return events.loc[events["event_type"] == "PTM"].copy()


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

    # Targets are intentionally limited to the next PTM inside the same lifecycle.
    # This keeps the first experiments aligned with your lifecycle concept.
    by_lifecycle = ptm.groupby("Lifecycle_ID", sort=False)
    ptm["target_next_URA"] = by_lifecycle["URA"].shift(-1)
    ptm["target_next_IRI"] = by_lifecycle["IRI"].shift(-1)
    ptm["target_next_event_date"] = by_lifecycle["event_date"].shift(-1)
    ptm["target_next_measurement_idx"] = by_lifecycle["Measurement_Idx"].shift(-1)
    ptm["target_horizon_days"] = (
        pd.to_datetime(ptm["target_next_event_date"]) - pd.to_datetime(ptm["event_date"])
    ).dt.days
    ptm["target_horizon_years"] = ptm["target_horizon_days"] / 365.25

    # Useful simple baselines to compare against immediately.
    ptm["baseline_persist_URA"] = ptm["URA"]
    ptm["baseline_persist_IRI"] = ptm["IRI"]

    # Lifecycle-relative position features make early models easier to interpret.
    lifecycle_size = by_lifecycle["Measurement_Idx"].transform("max")
    ptm["measurements_remaining_in_lifecycle"] = lifecycle_size - ptm["Measurement_Idx"]
    ptm["is_first_measurement_in_lifecycle"] = ptm["Measurement_Idx"].eq(1)
    ptm["has_previous_measurement_in_lifecycle"] = ptm["Measurement_Idx"].gt(1)
    ptm["has_next_measurement_in_lifecycle"] = ptm["target_next_URA"].notna()

    preferred_cols = (
        IDENTIFIER_COLUMNS
        + available_columns(ptm, STATIC_FEATURES)
        + available_columns(ptm, CURRENT_STATE_FEATURES)
        + available_columns(ptm, LAG_FEATURES)
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

    model_df = ptm.loc[:, preferred_cols].copy()

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
        print("  selected dtypes:")
        dtype_cols = [
            "Toim_lk",
            "Nopeus",
            "Pituus",
            "Measurement_Idx",
            "has_TP_interval",
            "target_next_URA",
            "target_next_IRI",
        ]
        available = [c for c in dtype_cols if c in df.columns]
        for col in available:
            print(f"    {col}: {df[col].dtype}")


def main() -> None:
    ensure_exists(IN_EVENT_HISTORY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading canonical event history PTM rows: {IN_EVENT_HISTORY}")
    ptm_rows = read_ptm_source_rows(IN_EVENT_HISTORY)

    print("Building PTM modeling dataset with lifecycle-aware next-measurement targets")
    model_df = build_model_table(ptm_rows)

    out_path = OUT_DIR / OUT_FILE
    model_df.to_parquet(out_path, index=False)

    print(f"Saved model dataset: {out_path}")
    print_summary(model_df)


if __name__ == "__main__":
    main()

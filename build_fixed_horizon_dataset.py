#!/usr/bin/env python3
"""
build_fixed_horizon_dataset.py

Build a fixed-horizon modeling dataset directly from the canonical modeling parquet.

Design:
    - one row = one PTM measurement event
    - targets are direct forecasts to 1, 2, 3, and 4 years ahead
    - target lookup stays inside the same lifecycle
    - the closest PTM observation to each target horizon is used if it falls
      within the configured tolerance window

This file is intended to support direct-horizon models where the split is done
later in the notebook or training script based on event time.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable, List

import numpy as np
import pandas as pd


IN_FILE = Path("data/road_model_dataset_v2.parquet")
OUT_DIR = Path("data")
OUT_FILE = Path("road_model_dataset_fixed_horizon_v2.parquet")
SUMMARY_FILE = Path("road_model_dataset_fixed_horizon_coverage_v2.csv")

HISTORY_YEARS = [1, 2, 3]
FORECAST_YEARS = [1, 2, 3, 4]
TIME_WINDOW_YEARS = 0.25

BASE_COLUMNS = [
    "Segment_ID",
    "Lifecycle_ID",
    "event_date",
    "year",
    "cycle_num",
    "KVL",
    "KVL_raskas",
    "KVL_kaista",
    "Nopeus",
    "Toim_lk",
    "Pituus",
    "IRI",
    "URA",
    "Pavement_Age_years",
    "Initial_URA",
    "Measurement_Idx",
    "Minor_TP_Count",
    "tp_count_interval",
    "has_TP_interval",
    "prev_IRI",
    "prev_URA",
    "Delta_t_days",
    "Delta_t_years",
    "tp_surface_type",
    "tp_material_type",
    "measurements_remaining_in_lifecycle",
    "is_first_measurement_in_lifecycle",
    "has_previous_measurement_in_lifecycle",
    "has_next_measurement_in_lifecycle",
    "baseline_persist_URA",
    "baseline_persist_IRI",
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


def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def nearest_index_for_queries(
    dates_ns: np.ndarray,
    query_ns: np.ndarray,
    tolerance_ns: int,
) -> np.ndarray:
    pos = np.searchsorted(dates_ns, query_ns, side="left")
    out = np.full(len(query_ns), -1, dtype=np.int64)

    left = pos - 1
    right = pos

    left_valid = left >= 0
    right_valid = right < len(dates_ns)

    left_diff = np.full(len(query_ns), np.inf)
    right_diff = np.full(len(query_ns), np.inf)

    left_diff[left_valid] = np.abs(query_ns[left_valid] - dates_ns[left[left_valid]])
    right_diff[right_valid] = np.abs(dates_ns[right[right_valid]] - query_ns[right_valid])

    choose_left = left_diff <= right_diff
    chosen = np.where(choose_left, left, right)
    chosen_diff = np.minimum(left_diff, right_diff)

    valid = chosen_diff <= tolerance_ns
    out[valid] = chosen[valid]
    return out


def add_time_history_and_targets(
    frame: pd.DataFrame,
    history_years: List[int],
    forecast_years: List[int],
    time_window_years: float,
) -> pd.DataFrame:
    ns_per_year = int(round(365.25 * 24 * 60 * 60 * 1_000_000_000))
    tolerance_ns = int(round(time_window_years * 365.25 * 24 * 60 * 60 * 1_000_000_000))

    frame = frame.sort_values(["Lifecycle_ID", "event_date"], kind="mergesort").reset_index(drop=True)
    row_count = len(frame)
    if row_count == 0:
        return frame

    lifecycle_ids = frame["Lifecycle_ID"].to_numpy()
    dates_ns_all = frame["event_date"].astype("int64").to_numpy()
    iri_all = frame["IRI"].to_numpy(dtype=float)
    ura_all = frame["URA"].to_numpy(dtype=float)

    group_starts = np.flatnonzero(np.r_[True, lifecycle_ids[1:] != lifecycle_ids[:-1]])
    group_ends = np.r_[group_starts[1:], row_count]
    total_groups = len(group_starts)
    start_time = perf_counter()

    output_arrays = {}
    for lag in history_years:
        output_arrays[f"IRI_hist_{lag}y"] = np.full(row_count, np.nan)
        output_arrays[f"URA_hist_{lag}y"] = np.full(row_count, np.nan)
        output_arrays[f"history_gap_years_{lag}y"] = np.full(row_count, np.nan)

    for horizon in forecast_years:
        output_arrays[f"target_IRI_{horizon}y"] = np.full(row_count, np.nan)
        output_arrays[f"target_URA_{horizon}y"] = np.full(row_count, np.nan)
        output_arrays[f"actual_horizon_years_{horizon}y"] = np.full(row_count, np.nan)

    for group_idx, (start, end) in enumerate(zip(group_starts, group_ends), start=1):
        dates_ns = dates_ns_all[start:end]
        iri = iri_all[start:end]
        ura = ura_all[start:end]
        target_slice = slice(start, end)

        for lag in history_years:
            query_ns = dates_ns - lag * ns_per_year
            idx = nearest_index_for_queries(dates_ns, query_ns, tolerance_ns)
            valid = idx >= 0

            output_arrays[f"IRI_hist_{lag}y"][target_slice] = np.where(valid, iri[idx], np.nan)
            output_arrays[f"URA_hist_{lag}y"][target_slice] = np.where(valid, ura[idx], np.nan)
            output_arrays[f"history_gap_years_{lag}y"][target_slice] = np.where(
                valid, (dates_ns - dates_ns[idx]) / ns_per_year, np.nan
            )

        for horizon in forecast_years:
            query_ns = dates_ns + horizon * ns_per_year
            idx = nearest_index_for_queries(dates_ns, query_ns, tolerance_ns)
            valid = idx >= 0

            output_arrays[f"target_IRI_{horizon}y"][target_slice] = np.where(valid, iri[idx], np.nan)
            output_arrays[f"target_URA_{horizon}y"][target_slice] = np.where(valid, ura[idx], np.nan)
            output_arrays[f"actual_horizon_years_{horizon}y"][target_slice] = np.where(
                valid, (dates_ns[idx] - dates_ns) / ns_per_year, np.nan
            )

        if group_idx == 1 or group_idx % 50_000 == 0 or group_idx == total_groups:
            elapsed_s = perf_counter() - start_time
            rate = group_idx / elapsed_s if elapsed_s > 0 else float("inf")
            remaining_groups = total_groups - group_idx
            eta_s = remaining_groups / rate if rate > 0 else float("nan")
            print(
                "Progress: "
                f"{group_idx:,}/{total_groups:,} lifecycles "
                f"({group_idx / total_groups:.1%}), "
                f"elapsed={elapsed_s / 60:.1f} min, "
                f"ETA={eta_s / 60:.1f} min"
            )

    for column_name, values in output_arrays.items():
        frame[column_name] = values

    return frame


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in FORECAST_YEARS:
        rows.append(
            {
                "horizon_years": horizon,
                "IRI_target_share": frame[f"target_IRI_{horizon}y"].notna().mean(),
                "URA_target_share": frame[f"target_URA_{horizon}y"].notna().mean(),
                "mean_actual_horizon_years": frame[f"actual_horizon_years_{horizon}y"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_exists(IN_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading modeling dataset: {IN_FILE}")
    df = pd.read_parquet(IN_FILE, columns=BASE_COLUMNS)
    ensure_required_columns(df, ["Lifecycle_ID", "event_date", "IRI", "URA"])

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["Lifecycle_ID", "event_date"]).copy()

    print(f"Building fixed-horizon targets for rows={len(df):,}")
    out_df = add_time_history_and_targets(
        df,
        history_years=HISTORY_YEARS,
        forecast_years=FORECAST_YEARS,
        time_window_years=TIME_WINDOW_YEARS,
    )

    out_path = OUT_DIR / OUT_FILE
    out_df.to_parquet(out_path, index=False)
    print(f"Saved fixed-horizon dataset: {out_path}")

    summary = coverage_summary(out_df)
    summary_path = OUT_DIR / SUMMARY_FILE
    summary.to_csv(summary_path, index=False)
    print(f"Saved coverage summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

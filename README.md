# Vayla Road Deterioration Prediction

This repository contains code and notebooks for a road deterioration prediction project based on historical road condition, maintenance, and segment data.

The main indicators of interest are:

- `URA` for rutting
- `IRI` for roughness

The repository is organized around two main tasks:

- transforming raw historical road data into a structured event-history dataset
- building a modeling dataset and testing baseline prediction models

## Main Data Files

### `data/road_event_history_v2.parquet`

Canonical chronological event-history dataset.

Contains:

- PTM measurement events
- TP treatment events
- segment identifiers
- lifecycle identifiers
- previous-measurement fields
- treatment interval fields
- static segment attributes

This file is the main source dataset for downstream work.

### `data/road_model_dataset_v1.parquet`

Derived PTM-only modeling dataset created from the canonical event history.

Contains:

- one row per PTM measurement
- current features
- lag features
- lifecycle features
- static features
- next-measurement targets:
  - `target_next_URA`
  - `target_next_IRI`

## Scripts

### [filter_and_create_ml_data_v3.py](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/filter_and_create_ml_data_v3.py)

Builds the canonical event-history dataset from the raw wide-format parquet.

Main responsibilities:

- read the raw historical dataset row-group by row-group
- filter segments and invalid rows
- extract PTM and TP histories from indexed wide columns
- build one chronological event table
- compute transition, interval, and lifecycle fields
- save the final canonical parquet

Output:

- `data/road_event_history_v2.parquet`

### [build_model_dataset.py](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/build_model_dataset.py)

Builds the modeling dataset from the canonical event-history parquet.

Main responsibilities:

- read PTM rows from the canonical dataset
- preserve segment and lifecycle structure
- create next-measurement targets within the same lifecycle
- keep selected current, lag, lifecycle, and static variables
- save the final modeling parquet

Output:

- `data/road_model_dataset_v1.parquet`

## Notebooks

### [baseline_simple_models.ipynb](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/baseline_simple_models.ipynb)

Notebook for simple baseline experiments.

Includes:

- persistence baseline
- linear regression
- ridge regression
- feature ablation experiments
- coefficient inspection

### [random_forest_baseline.ipynb](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/random_forest_baseline.ipynb)

Notebook for random forest baseline experiments.

Includes:

- persistence baseline comparison
- random forest models with several feature sets
- feature importance inspection

## Documentation

### [dataset_initial_to_road_deterioration_transformation.md](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/dataset_initial_to_road_deterioration_transformation.md)

Detailed documentation of the canonical event-history dataset.

Describes:

- source data structure
- transformation logic
- filtering rules
- lifecycle logic
- important columns
- typical dtypes
- expected missing values

### [baseline_model_results_summary.md](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/baseline_model_results_summary.md)

Summary file for baseline model results and notes.

## Other Files

### `road_event_dataset_information_v1.ipynb`

Notebook for inspecting or exploring the road event dataset.

### `additional_data_transformation.ipynb`

Notebook related to earlier or additional transformation work.

### `filter_and_create_ml_data_v2.py`

Older version of the event-history data creation script.

## Recommended File Order

If rebuilding the pipeline from scratch, the main order is:

1. run [filter_and_create_ml_data_v3.py](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/filter_and_create_ml_data_v3.py)
2. run [build_model_dataset.py](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/build_model_dataset.py)
3. open [baseline_simple_models.ipynb](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/baseline_simple_models.ipynb)
4. open [random_forest_baseline.ipynb](/c:/Users/mkjun/Documents/School/misc_maisteri_kurssit/case_studies_operations_research/Vayla_projekti/random_forest_baseline.ipynb)

## Notes

- The canonical dataset is the source of truth for downstream datasets.
- The modeling dataset is a derived dataset for prediction experiments.
- `Toim_lk` is categorical and should remain categorical in downstream modeling.

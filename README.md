# Vayla Road Deterioration Prediction

This repository contains a classical machine-learning workflow for predicting road deterioration from historical road condition, maintenance, and segment data.

Main prediction targets:

- `URA`: rutting
- `IRI`: roughness

The current unified workflow is documented in detail in `methodology_cleanup_documentation.md`.

## Current Workflow

The recommended classical modeling path is:

1. Build the canonical event-history dataset.
2. Build the supervised PTM-level modeling dataset.
3. Run the controlled model-family comparison.
4. Tune the selected final model family.
5. Train and report the final selected model.

## Main Data Files

| File | Status | Purpose |
| --- | --- | --- |
| `data/road_event_history_v2.parquet` | Current canonical source | Chronological event-history table with PTM measurements, TP treatments, segment identifiers, lifecycle identifiers, treatment interval fields, and static road attributes. |
| `data/road_model_dataset_v2.parquet` | Current modeling dataset | Supervised PTM-level dataset used by the unified comparison, tuning, and final training notebooks. |
| `data/road_model_dataset_v1.parquet` | Older version | Earlier supervised modeling dataset used before the final methodology cleanup. |

## Core Scripts

| File | Status | Purpose |
| --- | --- | --- |
| `filter_and_create_ml_data_v3.py` | Current event-history builder | Builds `data/road_event_history_v2.parquet` from the raw wide-format source data. |
| `build_model_dataset_v2.py` | Current modeling-dataset builder | Builds `data/road_model_dataset_v2.parquet` from the canonical event-history dataset. |
| `build_model_dataset.py` | Older version | Earlier modeling-dataset builder for `road_model_dataset_v1.parquet`. |
| `filter_and_create_ml_data_v2.py` | Deprecated/test version | Older event-history construction script kept for reference. |

## Unified Modeling Notebooks

| File | Workflow stage | Purpose |
| --- | --- | --- |
| `model_comparison_final.ipynb` | Controlled model comparison | Compares persistence, linear/ridge, random forest, HistGradientBoosting, and GradientBoosting under one shared dataset, split, feature pipeline, and evaluation protocol. |
| `model_comparison_final_plots.ipynb` | Reporting/visualization | Additional plotting notebook for final comparison results. |
| `model_final_hyperparameter_tuning.ipynb` | Final model tuning | Tunes the final URA-focused model candidates, including HGB, sampled RF, delta-target HGB, and focused second-stage HGB delta search. |
| `model_final_training.ipynb` | Final model training/reporting | Loads the best tuning candidate, trains on train + validation, evaluates once on test, and saves final model/report artifacts. |

## Documentation

| File | Purpose |
| --- | --- |
| `methodology_cleanup_documentation.md` | Main reference for the unified workflow, including split design, feature engineering, model comparison, tuning, final training, and saved artifacts. |
| `methodology_cleanup_plan.md` | Planning document that defined the unified methodology before implementation. |
| `dataset_initial_to_road_deterioration_transformation.md` | Detailed documentation of the event-history data transformation. |
| `baseline_model_results_summary.md` | Summary of earlier baseline model experiments. Historical reference only. |

## Historical / Pre-Unification Notebooks

These notebooks were useful during exploration, but the final classical methodology is now represented by the unified notebooks above.

| File | Status |
| --- | --- |
| `simple_models_baseline.ipynb` | Early baseline experiment. Historical. |
| `simple_models_baseline_v2.ipynb` | Later simple-model baseline experiment. Superseded by `model_comparison_final.ipynb`. |
| `random_forest_baseline.ipynb` | Early random forest baseline. Historical. |
| `random_forest_baseline_v2.ipynb` | Later random forest baseline. Superseded by `model_comparison_final.ipynb`. |
| `hist_gradient_boosting_baseline_v2.ipynb` | HGB baseline exploration. Superseded by the unified comparison and tuning notebooks. |
| `road_event_dataset_information_v1.ipynb` | Event-dataset inspection notebook. Reference only. |
| `road_model_dataset_information.ipynb` | Modeling-dataset inspection notebook. Reference only. |

## Results Directories

| Directory | Produced by | Contents |
| --- | --- | --- |
| `results/final_model_comparison/` | `model_comparison_final.ipynb` | Validation/test comparison tables, predictions, direct-vs-delta checks, and comparison figures. |
| `results/final_model_tuning/` | `model_final_hyperparameter_tuning.ipynb` | URA-focused tuning tables, tuned test checks, tuning config, and tuning predictions. |
| `results/final_model_training/` | `model_final_training.ipynb` | Final selected model artifact, metadata, test results, residual outputs, comparison table, and final figures. |

## Recommended Run Order

For the current classical workflow:

1. Run `filter_and_create_ml_data_v3.py`.
2. Run `build_model_dataset_v2.py`.
3. Run `model_comparison_final.ipynb`.
4. Run `model_final_hyperparameter_tuning.ipynb`.
5. Run `model_final_training.ipynb`.

If only final model selection/reporting is needed and the data plus tuning outputs already exist, start from `model_final_training.ipynb`.

## Notes

- `data/road_event_history_v2.parquet` is the source of truth for downstream datasets.
- `data/road_model_dataset_v2.parquet` is the current supervised table for the unified classical workflow.
- The final comparison uses an ELY-stratified `Segment_ID` group split.
- `ELY` is used for splitting and reporting, not as a model input.
- `target_horizon_years` is included as a model input in every final feature mixture.
- The current final tabular feature mixture is `current_static_lag_lifecycle_material`.

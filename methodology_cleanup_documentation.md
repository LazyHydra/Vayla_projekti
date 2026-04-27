# Methodology Cleanup Documentation

This document summarizes the final classical machine-learning workflow for the Vayla road deterioration project. It is the concise reference for the unified methodology implemented in:

- `model_comparison_final.ipynb`
- `model_final_hyperparameter_tuning.ipynb`
- `model_final_training.ipynb`

Neural-network work is intentionally outside this document.

## Goal

Predict next-measurement road condition from historical road condition, maintenance, and segment data.

Main targets:

- `target_next_URA`: next rutting value
- `target_next_IRI`: next roughness value

The final selected classical model focuses on `target_next_URA`, because URA had more room for improvement and benefited clearly from delta-target modeling.

## Data

| File | Role |
| --- | --- |
| `data/road_event_history_v2.parquet` | Canonical chronological event-history table with PTM measurements, TP treatments, segment/lifecycle identifiers, and treatment information. |
| `data/road_model_dataset_v2.parquet` | Supervised PTM-level modeling table used for comparison, tuning, and final training. |

Each supervised row represents a current PTM measurement. The target is the next PTM measurement in the same inferred lifecycle.

## Split

All final classical notebooks use the same split logic:

- split unit: `Segment_ID`
- stratification: `ELY`
- train / validation / test: 70% / 15% / 15%
- random seed: `42`

Every row from a physical road segment is assigned to exactly one split. `ELY` is used for split balance and reporting, not as a model input.

This group split was chosen over a pure time split because the main final question is generalization to held-out road segments. Earlier time-split experiments mostly tested future rows from already-seen segments.

## Final Feature Set

The final tabular feature mixture is:

```text
current_static_lag_lifecycle_material
```

It contains:

- current condition: `URA`, `IRI`
- horizon: `target_horizon_years`
- static road attributes: `KVL`, `KVL_raskas`, `KVL_kaista`, `Nopeus`, `Toim_lk`
- lag/history: `prev_URA`, `prev_IRI`, `Delta_t_years`
- lifecycle context: `observed_lifecycle_age_years`, `Minor_TP_Count`
- recent treatment/material context: `tp_count_interval`, `surface_type_current`, `material_type_current`, `years_since_material_update`

`observed_lifecycle_age_years` is an alias of `Pavement_Age_years`. It means time since first PTM in the inferred lifecycle, not true pavement construction age.

## Current Material State

The final workflow derives carried-forward material state from `data/road_event_history_v2.parquet`, rather than using the older interval material fields as direct inputs.

Derived fields:

- `surface_type_current`
- `material_type_current`
- `years_since_material_update`

The notebooks load TP rows, classify raw `Tp_pinta` and `Tp_tyomen` values into coarse surface/material categories, then use a backward as-of merge by `Segment_ID` and event date. Same-day TP is allowed because the event history orders TP before PTM on the same date.

Older interval fields are not used as main final inputs:

- `tp_surface_type`
- `tp_material_type`

## Excluded Inputs

The final models avoid future-known fields and leakage-prone identifiers.

Excluded as model inputs:

- target-side fields such as `target_next_URA`, `target_next_IRI`, `target_next_event_date`, `target_next_measurement_idx`
- raw road identifiers such as `Tie`, `Aosa`, `Aet`, `Losa`, `Let`
- `ELY`
- `Measurement_Idx`
- `Initial_URA`
- `has_TP_interval`
- `is_major_reset`
- `is_phantom_reset`

## Preprocessing

Numeric features:

- median imputation
- scaling only for linear/ridge models
- no scaling for tree-based models

Categorical features:

- missing-value imputation
- one-hot encoding with `handle_unknown="ignore"`

Categorical inputs in the final feature set:

- `Toim_lk`
- `surface_type_current`
- `material_type_current`

Preprocessing is fit only on the relevant training data. In final training, the preprocessing pipeline is fit on the combined train + validation pool before the final test evaluation.

## Controlled Model Comparison

Implemented in `model_comparison_final.ipynb`.

Compared model families:

- persistence baseline
- linear regression
- ridge regression
- random forest
- `HistGradientBoostingRegressor`
- `GradientBoostingRegressor`

Metrics:

- MAE
- RMSE
- R2

Evaluation views:

- overall validation/test
- test by horizon window
- test by ELY

Horizon windows:

- 1-year: 274 to 457 days
- 2-year: 639 to 822 days
- 3-year: 1004 to 1187 days
- 4-year: 1370 to 1553 days

Main artifacts:

- `results/final_model_comparison/validation_results.csv`
- `results/final_model_comparison/test_results_overall.csv`
- `results/final_model_comparison/test_results_breakdowns.csv`
- `results/final_model_comparison/direct_vs_delta_test_results.csv`
- `results/final_model_comparison/test_predictions.parquet`
- `results/final_model_comparison/figures/`

## Comparison Results

The strongest untuned model in the controlled comparison was HGB with `current_static_lag_lifecycle_material`.

Test results for `target_next_URA`:

| Model | Target type | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: |
| HGB | direct | 1.1506 | 1.7652 | 0.8408 |
| Random forest | direct | 1.1411 | 1.7742 | 0.8391 |
| Gradient boosting | direct | 1.2093 | 1.8412 | 0.8268 |
| Linear regression | direct | 1.2864 | 1.9235 | 0.8109 |
| Ridge | direct | 1.2864 | 1.9235 | 0.8109 |
| Persistence | direct | 1.8945 | 2.6460 | 0.6422 |

Direct-vs-delta testing also showed that URA benefits from predicting change from current state:

| Target type | Test MAE | Test RMSE | Test R2 |
| --- | ---: | ---: | ---: |
| HGB direct | 1.1506 | 1.7652 | 0.8408 |
| HGB delta converted to actual | 1.1446 | 1.7542 | 0.8427 |

This motivated URA-focused HGB delta tuning.

## Hyperparameter Tuning

Implemented in `model_final_hyperparameter_tuning.ipynb`.

The tuning target is `target_next_URA`. The notebook first tunes direct HGB and sampled RF candidates, then evaluates HGB as a delta model, then runs a focused second-stage HGB delta search.

Important saved artifacts:

- `results/final_model_tuning/all_ura_validation_tuning_results.csv`
- `results/final_model_tuning/hgb_ura_validation_tuning.csv`
- `results/final_model_tuning/rf_ura_validation_tuning_sampled.csv`
- `results/final_model_tuning/hgb_ura_delta_validation_tuning.csv`
- `results/final_model_tuning/hgb_ura_delta_focused_tuning.csv`
- `results/final_model_tuning/hgb_ura_delta_all_tuning.csv`
- `results/final_model_tuning/ura_tuning_config.json`
- `model_final_hyperparameter_tuning_plots.ipynb` for visual summaries

First-pass best direct HGB validation result:

| Target type | Validation MAE | Validation RMSE | Validation R2 |
| --- | ---: | ---: | ---: |
| direct | 1.1103 | 1.7130 | 0.8495 |

Initial tuned HGB delta result:

| Target type | Validation MAE | Validation RMSE | Validation R2 |
| --- | ---: | ---: | ---: |
| delta converted to actual | 1.1047 | 1.7023 | 0.8514 |

Focused HGB delta search winner:

```text
model = HistGradientBoostingRegressor
target_type = delta_converted_to_actual
learning_rate = 0.10
max_iter = 400
max_leaf_nodes = 127
min_samples_leaf = 150
l2_regularization = 0.10
validation MAE = 1.0564
validation RMSE = 1.6454
validation R2 = 0.8612
```

This focused search was worthwhile: validation RMSE improved from `1.7023` to `1.6454`.

## Final Training

Implemented in `model_final_training.ipynb`.

The notebook selects the lowest-validation-RMSE candidate from the tuning outputs. With the current saved results, it selects the focused HGB delta winner.

Final training uses:

- fit pool: train + validation segments
- final evaluation: untouched test segments
- target representation: `delta_target_next_URA = target_next_URA - URA`
- reported prediction: `URA + raw_delta_prediction`

The HGB estimator is fit using the combined train + validation pool, with sklearn's internal `early_stopping=True` and `validation_fraction=0.1`. In the saved final model, HGB ran all `400` iterations, so early stopping did not truncate the selected `max_iter`.

## Final Model Result

Saved in:

- `results/final_model_training/final_ura_model_test_result.csv`

Final test result:

| Model | Target type | Train rows | Test rows | Test MAE | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HGB | delta converted to actual | 3,258,128 | 575,195 | 1.0578 | 1.6437 | 0.8619 |

The final test result is consistent with validation:

| Stage | RMSE |
| --- | ---: |
| Focused tuning validation | 1.6454 |
| Final test | 1.6437 |

Compared with the original controlled-comparison HGB test RMSE of `1.7652`, the final tuned delta HGB improves RMSE by about `6.9%`.

## Final Training Artifacts

Produced under `results/final_model_training/`:

| File | Purpose |
| --- | --- |
| `final_ura_model.joblib` | Serialized final fitted sklearn pipeline. |
| `final_ura_model_metadata.json` | Model name, target type, features, params, and random seed. |
| `final_ura_model_test_result.csv` | Overall final test metrics. |
| `final_ura_model_comparison_table.csv` | Final model compared with earlier comparison results. |
| `final_ura_model_test_breakdowns.csv` | Overall, horizon-window, and ELY breakdowns. |
| `final_ura_model_test_predictions.parquet` | Test predictions with raw delta and converted prediction. |
| `final_ura_model_test_predictions_with_residuals.parquet` | Predictions plus residuals and absolute errors. |
| `figures/` | Final plots for predicted-vs-actual, residuals, horizon RMSE, ELY RMSE, and comparison. |

## Recommended Run Order

For a clean rerun:

1. `filter_and_create_ml_data_v3.py`
2. `build_model_dataset_v2.py`
3. `model_comparison_final.ipynb`
4. `model_final_hyperparameter_tuning.ipynb`
5. `model_final_hyperparameter_tuning_plots.ipynb`
6. `model_final_training.ipynb`

If data and tuning outputs already exist, rerun only `model_final_training.ipynb` to reproduce the final model and report artifacts.

## Methodology Principles

- Use `road_event_history_v2.parquet` as the canonical historical source.
- Use `road_model_dataset_v2.parquet` as the supervised tabular modeling table.
- Use one ELY-stratified `Segment_ID` group split.
- Keep test data untouched until final reporting.
- Use validation RMSE for model and hyperparameter selection.
- Use `ELY` only for splitting and reporting.
- Use `target_horizon_years` as an explicit input.
- Use carried-forward current material state from TP history.
- Avoid target-side fields, future-known fields, and raw road identifiers as inputs.
- Report final performance overall, by horizon window, and by ELY.


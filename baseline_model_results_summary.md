# Baseline Model Results Summary

This file summarizes the first baseline experiments for predicting:

- `target_next_URA`
- `target_next_IRI`

Each row in the modeling dataset is one current `PTM` measurement, and the target is the next `PTM` measurement within the same lifecycle.

## Shared Feature Groups

The same feature groups were used in the ablation experiments for both linear models and random forest:

| Feature set | Included variables |
| --- | --- |
| `current_only` | `URA`, `IRI` |
| `current_plus_lag` | `URA`, `IRI`, `prev_URA`, `prev_IRI`, `Delta_t_years` |
| `current_lag_lifecycle` | current + lag + `Pavement_Age_years`, `Initial_URA`, `Measurement_Idx`, `Minor_TP_Count`, `tp_count_interval`, `has_TP_interval` |
| `current_lag_lifecycle_static` | current + lag + lifecycle + `KVL`, `KVL_raskas`, `KVL_kaista`, `Nopeus`, `Toim_lk` |

## Linear Regression and Ridge

### What was predicted

- `target_next_URA`
- `target_next_IRI`

### Main inputs used

- current condition: `URA`, `IRI`
- short history: `prev_URA`, `prev_IRI`
- time gap: `Delta_t_years`
- lifecycle context: `Pavement_Age_years`, `Initial_URA`, `Measurement_Idx`, `Minor_TP_Count`, `tp_count_interval`, `has_TP_interval`
- static segment context: `KVL`, `KVL_raskas`, `KVL_kaista`, `Nopeus`, `Toim_lk`

### Baseline vs best linear results

| Target | Persistence RMSE | Best linear RMSE | Best feature set | Best linear R2 |
| --- | ---: | ---: | --- | ---: |
| `target_next_IRI` | 0.408 | 0.375 | `current_lag_lifecycle_static` | 0.859 |
| `target_next_URA` | 2.714 | 1.814 | `current_lag_lifecycle_static` | 0.834 |

### Linear ablation results for `target_next_IRI`

| Feature set | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| `current_only` | `linear_regression` | 0.2241 | 0.3875 | 0.8492 |
| `current_only` | `ridge_alpha_1` | 0.2241 | 0.3875 | 0.8492 |
| `current_plus_lag` | `linear_regression` | 0.2168 | 0.3777 | 0.8568 |
| `current_plus_lag` | `ridge_alpha_1` | 0.2168 | 0.3777 | 0.8568 |
| `current_lag_lifecycle` | `linear_regression` | 0.2133 | 0.3762 | 0.8579 |
| `current_lag_lifecycle` | `ridge_alpha_1` | 0.2133 | 0.3762 | 0.8579 |
| `current_lag_lifecycle_static` | `linear_regression` | 0.2142 | 0.3754 | 0.8586 |
| `current_lag_lifecycle_static` | `ridge_alpha_1` | 0.2142 | 0.3753 | 0.8586 |

### Linear ablation results for `target_next_URA`

| Feature set | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| `current_only` | `linear_regression` | 1.3241 | 1.8683 | 0.8234 |
| `current_only` | `ridge_alpha_1` | 1.3241 | 1.8683 | 0.8234 |
| `current_plus_lag` | `linear_regression` | 1.3081 | 1.8417 | 0.8284 |
| `current_plus_lag` | `ridge_alpha_1` | 1.3081 | 1.8417 | 0.8284 |
| `current_lag_lifecycle` | `linear_regression` | 1.3142 | 1.8543 | 0.8260 |
| `current_lag_lifecycle` | `ridge_alpha_1` | 1.3142 | 1.8543 | 0.8260 |
| `current_lag_lifecycle_static` | `linear_regression` | 1.2761 | 1.8139 | 0.8335 |
| `current_lag_lifecycle_static` | `ridge_alpha_1` | 1.2761 | 1.8139 | 0.8335 |

### Main takeaways from linear models

- `current_only` is already strong for both targets
- adding lag features improves both `URA` and `IRI`
- lifecycle features help `URA` more clearly than `IRI`
- static features give the best overall result for both targets
- linear regression and ridge produced almost identical results

### Top linear coefficients for the best `target_next_URA` model

Best model:

- `linear_regression`
- feature set: `current_lag_lifecycle_static`

| Rank | Feature | Coefficient |
| ---: | --- | ---: |
| 1 | `num__URA` | 3.7340 |
| 2 | `num__KVL_kaista` | 0.4536 |
| 3 | `num__Measurement_Idx` | -0.2899 |
| 4 | `num__prev_URA` | 0.2665 |
| 5 | `num__tp_count_interval` | 0.2048 |
| 6 | `num__has_TP_interval` | -0.1994 |
| 7 | `num__Delta_t_years` | 0.1549 |
| 8 | `num__KVL` | -0.1397 |
| 9 | `cat__Toim_lk_seutu` | -0.1177 |
| 10 | `cat__Toim_lk_kanta` | 0.1129 |

### Top linear coefficients for the best `target_next_IRI` model

Best model:

- `linear_regression`
- feature set: `current_lag_lifecycle_static`

| Rank | Feature | Coefficient |
| ---: | --- | ---: |
| 1 | `num__IRI` | 0.8303 |
| 2 | `cat__Toim_lk_yhdys` | 0.1233 |
| 3 | `num__prev_IRI` | 0.1083 |
| 4 | `cat__Toim_lk_valta` | -0.0688 |
| 5 | `num__URA` | 0.0547 |
| 6 | `num__Measurement_Idx` | -0.0542 |
| 7 | `cat__Toim_lk_kanta` | -0.0479 |
| 8 | `num__Pavement_Age_years` | 0.0340 |
| 9 | `num__prev_URA` | -0.0192 |
| 10 | `num__KVL_kaista` | -0.0136 |

### Linear interpretation notes

- For `target_next_URA`, current `URA` is clearly the dominant predictor.
- For `target_next_IRI`, current `IRI` is clearly the dominant predictor.
- Numeric features were standardized before fitting, so numeric coefficient magnitudes are comparable.
- Categorical coefficients should be interpreted relative to the omitted reference category.

## Random Forest

### Random forest results for `target_next_IRI`

| Feature set | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| `persistence` | `persistence` | 0.2201 | 0.4083 | 0.8328 |
| `current_only` | `random_forest` | 0.2263 | 0.3975 | 0.8415 |
| `current_plus_lag` | `random_forest` | 0.2128 | 0.3748 | 0.8591 |
| `current_lag_lifecycle` | `random_forest` | 0.2119 | 0.3740 | 0.8597 |
| `current_lag_lifecycle_static` | `random_forest` | 0.2098 | 0.3711 | 0.8619 |

### Random forest results for `target_next_URA`

| Feature set | Model | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| `persistence` | `persistence` | 2.1364 | 2.7148 | 0.6259 |
| `current_only` | `random_forest` | 1.4010 | 1.9398 | 0.8090 |
| `current_plus_lag` | `random_forest` | 1.3531 | 1.8689 | 0.8227 |
| `current_lag_lifecycle` | `random_forest` | 1.3136 | 1.8233 | 0.8313 |
| `current_lag_lifecycle_static` | `random_forest` | 1.2706 | 1.7802 | 0.8392 |

### Random forest vs best linear model

| Target | Best linear RMSE | Best RF RMSE | Better model |
| --- | ---: | ---: | --- |
| `target_next_IRI` | 0.3754 | 0.3711 | `random_forest` |
| `target_next_URA` | 1.8139 | 1.7802 | `random_forest` |

### Top random forest importances for the best `target_next_URA` model

Best model:

- `random_forest`
- feature set: `current_lag_lifecycle_static`

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `num__URA` | 0.8638 |
| 2 | `num__IRI` | 0.0183 |
| 3 | `num__KVL_kaista` | 0.0177 |
| 4 | `num__KVL_raskas` | 0.0170 |
| 5 | `num__prev_URA` | 0.0154 |
| 6 | `num__KVL` | 0.0146 |
| 7 | `num__Delta_t_years` | 0.0115 |
| 8 | `num__Initial_URA` | 0.0111 |
| 9 | `num__Pavement_Age_years` | 0.0089 |
| 10 | `num__prev_IRI` | 0.0074 |

### Top random forest importances for the best `target_next_IRI` model

Best model:

- `random_forest`
- feature set: `current_lag_lifecycle_static`

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `num__IRI` | 0.8914 |
| 2 | `num__KVL_raskas` | 0.0162 |
| 3 | `num__prev_IRI` | 0.0161 |
| 4 | `num__URA` | 0.0137 |
| 5 | `num__KVL` | 0.0122 |
| 6 | `num__Initial_URA` | 0.0108 |
| 7 | `num__KVL_kaista` | 0.0088 |
| 8 | `num__Delta_t_years` | 0.0081 |
| 9 | `num__Pavement_Age_years` | 0.0063 |
| 10 | `num__prev_URA` | 0.0057 |

### Main takeaways from random forest

- For both `URA` and `IRI`, `current_lag_lifecycle_static` is the best feature set.
- Performance improves as more context is added.
- Random forest improves over persistence for both targets.
- Random forest also improves slightly over the best linear model for both targets.
- Current condition remains by far the strongest predictor in both targets.

## Overall Summary

- Both targets are strongly autoregressive.
- `IRI` behaves more like a short-memory prediction task.
- `URA` benefits more from lifecycle and static context.
- Static and lifecycle features help, but most of the predictive power still comes from current condition.

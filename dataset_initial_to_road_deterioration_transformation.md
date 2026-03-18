# Road Event History Dataset Documentation

This document describes the transformed road event-history dataset used for machine learning, forecasting, and pavement lifecycle analysis.

The dataset converts wide historical road data into **one chronological event table**, where both condition measurements and treatment events are preserved in time order.

**Output file:** `road_event_history_v2.parquet`

---

## 1. Purpose

The dataset is designed as a **canonical event-history source** for downstream analysis. Typical uses include:

- pavement condition prediction
    
- multi-step forecasting at future measurement dates
    
- lifecycle and deterioration analysis
    
- sequence modeling with measurements and treatments
    
- feature engineering from a single standardized source
    

It is **not** intended to be the final ML design matrix. Model-specific filtering and target construction should be done separately.

---

## 2. Source Data

### Source file

`historiadata_ALL.parquet`

### Raw structure

In the raw source:

- **1 row = 1 road segment**
    
- historical PTM measurements are stored in indexed columns such as `PTM_pvm_1`, `Iri_1`, `Ura_max_1`
    
- historical TP treatments are stored in indexed columns such as `Tp_pvm_1`, `Tp_pinta_1`, `Tp_työmen_1`
    
- the history is therefore in **wide format**, not event rows
    

### Segment key

The segment is identified by these columns:

- `ELY`
    
- `Tie`
    
- `Ajorata`
    
- `Kaista`
    
- `Aosa`
    
- `Aet`
    
- `Losa`
    
- `Let`
    

A derived segment identifier is created as:

`Segment_ID = "_".join(KEY_COLS as strings)`

### Static segment attributes

These are copied onto event rows when present in the source:

- `KVL`
    
- `KVL_raskas`
    
- `KVL_kaista`
    
- `Nopeus`
    
- `Toim_lk`
    
- `Pituus`
    

---

## 3. Why the Transformation Is Needed

The raw format is difficult to use directly because:

- time history is stored in columns instead of rows
    
- the number of historical observations varies by segment
    
- PTM measurements and TP treatments are separate indexed histories
    
- temporal ordering and interval logic are hard to compute in wide format
    
- ML pipelines need event rows, not sparse wide slots
    

The transformation solves this by producing one time-ordered dataset where each row is a single event.

---

## 4. Final Dataset Concept

### Row definition

**1 row = 1 event for 1 segment on 1 date**

### Event types

- `PTM` = measurement event
    
- `TP` = treatment event
    

### Event ordering

Rows are sorted by:

1. `Segment_ID`
    
2. `event_date`
    
3. `event_order`
    
4. `ptm_idx` / `tp_idx`
    

### Same-day rule

If a treatment and a measurement occur on the same date:

- `TP` is ordered first
    
- `PTM` is ordered second
    

This is implemented with:

- `event_order = 0` for `TP`
    
- `event_order = 1` for `PTM`
    

---

## 5. Transformation Pipeline

The pipeline has two main stages plus final consolidation.

### Stage 1: Streaming extraction from wide parquet

The raw parquet is read row-group by row-group to keep memory usage bounded.

For each chunk:

- filter segments to 100 m
    
- extract PTM events from indexed PTM columns
    
- extract TP events from indexed TP columns
    
- write extracted rows into temporary hash buckets by segment
    

This stage is where the strict ML-oriented filtering is applied.

### Stage 2: Build event history bucket by bucket

For each bucket:

- clean and deduplicate PTM rows
    
- clean and deduplicate TP rows
    
- compute PTM transition fields
    
- count TP events between measurement intervals
    
- infer lifecycle resets from PTM behavior
    
- attach surrounding PTM context to TP rows
    
- merge PTM and TP back into one event chronology
    

### Stage 3: Final consolidation

All processed buckets are merged into a single parquet file:

`road_event_history_v2.parquet`

---

## 6. Filtering and Cleaning Rules

## Segment filter

Only segments with:

`Pituus == 100`

are kept.

This filter is applied early in Stage 1 for memory efficiency.

## Date filters

Only events from **2005-01-01 onward** are retained.

Applied to:

- PTM dates
    
- TP dates
    

Code constants:

- `PTM_MIN_VALID_DATE = 2005-01-01`
    
- `TP_MIN_VALID_DATE = 2005-01-01`
    
- `MIN_VALID_PTM_YEAR = 2005`
    
- `MIN_VALID_TP_YEAR = 2005`
    

## PTM value cleaning

Before ML filtering, PTM numeric values are cleaned using broad validity bounds:

- `IRI` valid range: `0 <= IRI <= 20`
    
- `URA` valid range: `0 <= URA <= 80`
    

Values outside those ranges are set to missing.

## PTM ML filter

After cleaning, PTM rows are kept only if all of the following hold:

- `IRI` is not missing
    
- `URA` is not missing
    
- `IRI <= 10`
    
- `URA <= 40`
    

This means the final PTM rows in the dataset have:

- no missing `IRI`
    
- no missing `URA`
    
- bounded roughness and rutting values
    
- dates from 2005 onward
    
- 100 m segment length
    

## TP date cleaning

TP dates are cleaned as follows:

- year `1900` is treated as a placeholder and removed
    
- dates before `2005-01-01` are removed
    
- dates with year below `MIN_VALID_TP_YEAR` are removed
    

## PTM date cleaning

PTM dates are cleaned as follows:

- invalid dates are removed
    
- dates with year below `MIN_VALID_PTM_YEAR` are removed
    

---

## 7. Deduplication Rules

Deduplication is done separately for PTM and TP rows.

### PTM duplicate key

PTM rows are deduplicated by:

- segment key
    
- `event_date`
    
- `event_type`
    
- `ptm_idx`
    

### TP duplicate key

TP rows are deduplicated by:

- segment key
    
- `event_date`
    
- `event_type`
    
- `tp_idx`
    
- `Tp_pinta` if present
    
- `Tp_tyomen` if present
    

This allows multiple TP events on the same date when their indexed slot or metadata differs.

---

## 8. Derived Fields for PTM Rows

The following fields are computed from the ordered PTM sequence within each segment.

### Measurement linkage

- `prev_meas_date`: date of previous PTM measurement on the same segment
    
- `next_meas_date`: date of next PTM measurement on the same segment
    

### Previous values

- `prev_IRI`: previous PTM roughness value
    
- `prev_URA`: previous PTM rutting value
    

### Changes from previous PTM

- `delta_IRI = IRI - prev_IRI`
    
- `delta_URA = URA - prev_URA`
    

### Time gap since previous PTM

- `Delta_t_days`: days since previous PTM
    
- `Delta_t_years`: years since previous PTM, computed as `Delta_t_days / 365.25`
    

For the first PTM of a segment, these are missing:

- `prev_meas_date`
    
- `prev_IRI`
    
- `prev_URA`
    
- `delta_IRI`
    
- `delta_URA`
    
- `Delta_t_days`
    
- `Delta_t_years`
    

### TP counts between measurements

For each PTM row, treatment events are counted in the interval:

`(prev_meas_date, current_meas_date]`

This means:

- the lower bound is open
    
- the current PTM date is included
    
- same-day TP is counted for the current PTM
    

Derived fields:

- `tp_count_interval`: number of TP events in that interval
    
- `has_TP_interval`: whether `tp_count_interval > 0`
    

### Convenience copies for chronology

For PTM rows, the code also sets:

- `days_since_prev_meas = Delta_t_days`
    
- `days_until_next_meas = next_meas_date - event_date` in days
    

---

## 9. Lifecycle Logic

Lifecycle inference is heuristic and based on changes in `URA` between PTM measurements.

### Reset thresholds

Code constants:

- `KNOWN_TP_RESET_DROP_MM = -1.0`
    
- `PHANTOM_RESET_DROP_MM = -3.0`
    

### Major reset

A PTM row is flagged as a major reset if either of these is true:

1. there was at least one TP in the interval and `delta_URA <= -1`
    
2. there was no TP in the interval and `delta_URA <= -3`
    

Derived field:

- `is_major_reset`
    

### Phantom reset

A phantom reset is a major reset inferred **without a recorded TP**:

- `has_TP_interval == False`
    
- `delta_URA <= -3`
    

Derived field:

- `is_phantom_reset`
    

### Minor treatment

A PTM row is flagged as minor treatment if:

- there was at least one TP in the interval
    
- but the row is **not** a major reset
    

Derived field:

- `is_minor_treatment`
    

### Lifecycle indexing

Within each segment:

- `cycle_num` is the cumulative count of major resets
    
- `Lifecycle_ID = Segment_ID + "_C" + cycle_num`
    

Note that lifecycle numbering starts from the initial state of the segment and increases whenever a major reset is detected.

### Lifecycle age and position

Within each lifecycle:

- `Measurement_Idx`: running PTM index within lifecycle
    
- `Pavement_Age_years`: years since first PTM in the lifecycle
    
- `Initial_URA`: first `URA` value in the lifecycle
    
- `Minor_TP_Count`: cumulative count of PTM rows flagged as `is_minor_treatment` within the lifecycle
    

---

## 10. TP Context Fields

TP rows do not have their own measured `IRI` or `URA`, so they are enriched with context from surrounding PTM rows.

For each TP row, the code finds:

- the nearest previous PTM on the same segment
    
- the nearest next PTM on the same segment
    

### Time context

- `prev_meas_date`: previous PTM date before the TP
    
- `next_meas_date`: next PTM date after the TP
    
- `days_since_prev_meas`: days from previous PTM to TP
    
- `days_until_next_meas`: days from TP to next PTM
    

### Lifecycle context

If a previous PTM exists, the TP row inherits from that PTM:

- `Lifecycle_ID`
    
- `cycle_num`
    
- `Initial_URA`
    

`Pavement_Age_years` for TP is then computed as:

- time since the start date of that inherited lifecycle
    

### Fields intentionally missing on TP rows

TP rows do **not** have PTM transition values, so these remain missing:

- `IRI`
    
- `URA`
    
- `prev_IRI`
    
- `prev_URA`
    
- `delta_IRI`
    
- `delta_URA`
    
- `Delta_t_days`
    
- `Delta_t_years`
    

Also, interval-based PTM flags are not meaningfully defined for TP rows, so fields such as the following are initialized as missing on TP rows:

- `tp_count_interval`
    
- `has_TP_interval`
    
- `is_minor_treatment`
    
- `Minor_TP_Count`
    
- `is_major_reset`
    
- `is_phantom_reset`
    
- `Measurement_Idx`
    

If a TP occurs before the first valid PTM of a segment, lifecycle-related fields may remain missing.

---

## 11. Final Dataset Structure

### File

`road_event_history_v2.parquet`

### Event granularity

- one row per event
    
- event may be either PTM or TP
    

### Main identifiers

- `Segment_ID`: unique segment identifier
    
- `Lifecycle_ID`: derived lifecycle identifier within segment
    
- `Event_Idx`: running event index within segment chronology
    

### Ordering columns

- `event_date`
    
- `event_order`
    
- `ptm_idx`
    
- `tp_idx`
    

---

## 12. Important Columns

## Core chronology

- `event_date`: event date
    
- `year`: calendar year extracted from `event_date`
    
- `event_type`: `PTM` or `TP`
    
- `event_order`: same-day precedence, `TP=0`, `PTM=1`
    
- `Event_Idx`: running event number within segment
    

## Original wide-slot indices

- `ptm_idx`: original PTM slot number from the wide source
    
- `tp_idx`: original TP slot number from the wide source
    

These are useful for traceability back to the raw wide layout.

## PTM measurement fields

- `IRI`: measured roughness value
    
- `URA`: measured rutting value
    

## Treatment metadata

- `Tp_pinta`: treatment surface/type descriptor from source
    
- `Tp_tyomen`: treatment method descriptor from source
    

The code supports both source spellings:

- `Tp_työmen_i`
    
- `Tp_tyomen_i`
    

but stores the output in the normalized column:

- `Tp_tyomen`
    

## Transition fields

- `prev_IRI`
    
- `prev_URA`
    
- `delta_IRI`
    
- `delta_URA`
    
- `Delta_t_days`
    
- `Delta_t_years`
    

## Interval treatment fields

- `tp_count_interval`
    
- `has_TP_interval`
    

## Lifecycle fields

- `cycle_num`
    
- `Lifecycle_ID`
    
- `Measurement_Idx`
    
- `Pavement_Age_years`
    
- `Initial_URA`
    
- `Minor_TP_Count`
    
- `is_major_reset`
    
- `is_phantom_reset`
    
- `is_minor_treatment`
    

## Static segment attributes

- `KVL`
    
- `KVL_raskas`
    
- `KVL_kaista`
    
- `Nopeus`
    
- `Toim_lk`
    
- `Pituus`
    

---

## 13. Missing Values

Missing values are expected in several places.

### Expected missing on TP rows

Normally missing on TP rows:

- `IRI`
    
- `URA`
    
- `prev_IRI`
    
- `prev_URA`
    
- `delta_IRI`
    
- `delta_URA`
    
- `Delta_t_days`
    
- `Delta_t_years`
    
- `Measurement_Idx`
    

Some lifecycle fields may also be missing if no prior PTM exists.

### Expected missing on first PTM row of a segment

For the first PTM measurement of a segment:

- `prev_meas_date`
    
- `prev_IRI`
    
- `prev_URA`
    
- `delta_IRI`
    
- `delta_URA`
    
- `Delta_t_days`
    
- `Delta_t_years`
    

are missing by definition.

---

## 14. Differences from Older Versions

Compared with a measurement-only panel, this dataset:

- preserves full event chronology
    
- keeps TP rows as explicit events
    
- retains treatment metadata
    
- computes measurement transitions directly
    
- adds lifecycle inference
    
- applies stricter ML-oriented filtering during extraction
    

This makes it more suitable as a reusable base dataset for many tasks.

---

## 15. Recommended Usage

For analysis, always sort by:

- `Segment_ID`
    
- `event_date`
    
- `event_order`
    
- optionally `ptm_idx`, `tp_idx`
    

Typical target setup:

- predict the next PTM measurement
    
- use prior PTM and TP history as input
    
- optionally restrict modeling to PTM rows only
    

Common task-specific subsets:

- **PTM-only modeling:** keep only `event_type == "PTM"`
    
- **full event history modeling:** keep both PTM and TP
    
- **lifecycle analysis:** group by `Lifecycle_ID`
    
- **segment history analysis:** group by `Segment_ID`
    

Treat this dataset as the canonical history source, then derive task-specific training tables separately.

---

## 16. Important Assumptions

- reset detection is heuristic, not a ground-truth reconstruction
    
- static traffic variables may not reflect time-varying traffic over history
    
- source treatment records may be incomplete
    
- some wide slots may be sparse or inconsistent
    
- TP-only segments are dropped if no valid PTM remains after filtering
    

That last point comes from the code: after Stage 2, TP rows are kept only for segments that still contain valid PTM rows.

---

## 17. Quick Workflow

1. load `road_event_history_v2.parquet`
    
2. sort by segment and chronology
    
3. inspect a few segments manually
    
4. decide whether to model PTM rows only or full event history
    
5. define targets and features separately from this canonical table
    

Useful validation checks:

- TP appears before PTM on the same day
    
- `tp_count_interval` matches treatment history between PTMs
    
- reset flags look plausible for large `URA` drops
    
- lifecycle age resets at major reset points
    

---

## 18. Summary

`road_event_history_v2.parquet` is a unified chronological road history dataset where:

- PTM measurements are explicit rows
    
- TP treatments are explicit rows
    
- chronology is preserved
    
- transition fields are computed from measurement history
    
- lifecycle structure is inferred from `URA` behavior
    
- filtering is applied early for ML-oriented use
    

The current version specifically adds:

- 100 m segment filtering
    
- 2005+ event filtering
    
- strict PTM validity requirements
    
- bounded-memory streaming extraction
    

It should be treated as a **canonical event-history source**, not as a ready-made final ML table.

---

## 19. Column Cheat Sheet

| Column                 | Meaning                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| `Segment_ID`           | Derived unique identifier built from the segment key columns              |
| `Lifecycle_ID`         | Derived lifecycle identifier within a segment                             |
| `Event_Idx`            | Running event number within a segment after chronological sorting         |
| `event_date`           | Date of the event                                                         |
| `year`                 | Calendar year extracted from `event_date`                                 |
| `event_type`           | Event type: `PTM` or `TP`                                                 |
| `event_order`          | Same-day ordering: `TP=0`, `PTM=1`                                        |
| `ptm_idx`              | Original PTM slot index from the wide source                              |
| `tp_idx`               | Original TP slot index from the wide source                               |
| `IRI`                  | Measured roughness value on PTM rows                                      |
| `URA`                  | Measured rutting value on PTM rows                                        |
| `Tp_pinta`             | Treatment surface/type descriptor                                         |
| `Tp_tyomen`            | Treatment method descriptor                                               |
| `prev_meas_date`       | Previous PTM date on the same segment                                     |
| `next_meas_date`       | Next PTM date on the same segment or surrounding a TP                     |
| `prev_IRI`             | Previous PTM `IRI` value                                                  |
| `prev_URA`             | Previous PTM `URA` value                                                  |
| `delta_IRI`            | Change in `IRI` since previous PTM                                        |
| `delta_URA`            | Change in `URA` since previous PTM                                        |
| `Delta_t_days`         | Days since previous PTM                                                   |
| `Delta_t_years`        | Years since previous PTM                                                  |
| `days_since_prev_meas` | Days from previous PTM to current event                                   |
| `days_until_next_meas` | Days from current event to next PTM                                       |
| `tp_count_interval`    | Number of TP events in `(prev_meas_date, current_meas_date]` for PTM rows |
| `has_TP_interval`      | Whether at least one TP occurred in that PTM interval                     |
| `is_major_reset`       | Heuristic lifecycle reset flag                                            |
| `is_phantom_reset`     | Reset inferred without recorded TP                                        |
| `is_minor_treatment`   | TP interval without major reset                                           |
| `cycle_num`            | Lifecycle index within a segment                                          |
| `Measurement_Idx`      | PTM sequence number within lifecycle                                      |
| `Pavement_Age_years`   | Years since lifecycle start                                               |
| `Initial_URA`          | First `URA` observed in the lifecycle                                     |
| `Minor_TP_Count`       | Cumulative count of minor-treatment PTM rows within lifecycle             |
| `KVL`                  | Segment traffic attribute                                                 |
| `KVL_raskas`           | Heavy traffic attribute                                                   |
| `KVL_kaista`           | Lane-level traffic attribute                                              |
| `Nopeus`               | Speed-related segment attribute                                           |
| `Toim_lk`              | Functional class attribute                                                |
| `Pituus`               | Segment length                                                            |

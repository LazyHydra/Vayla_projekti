# Road Event History Dataset Documentation

This document explains the purpose, structure, transformation logic, and usage of the unified road event history dataset created from the raw road deterioration history data.

It is written for engineers and data scientists who need to understand:

- what the original source data looked like
- what was changed during transformation
- what information is preserved in the final dataset
- what assumptions are embedded in the dataset
- how the resulting dataset should be used for modeling and analysis

---

# 1. Purpose of the Dataset

The goal of the transformation is to convert a wide historical road dataset into a single chronological event-history dataset that is flexible enough for machine learning, analytics, and feature engineering.

The final dataset is designed to support use cases such as:

- predicting pavement condition at future measurement events
- predicting one or multiple measurements ahead
- modeling deterioration within one pavement lifecycle
- modeling deterioration across multiple lifecycles
- using full history or partial history as model input
- encoding maintenance and restoration events explicitly in the input history
- deriving multiple downstream modeling datasets from one canonical source

A key design goal was to avoid locking the project too early into one specific modeling setup.

Instead of producing multiple specialized datasets, the transformation produces **one canonical parquet file** that preserves the most important historical information in a normalized and chronological structure.

---

# 2. Original Dataset

## 2.1 Source file

The original source dataset is:

`historiadata_ALL.parquet`

It is a unified historical road dataset where each row represents one road segment and historical measurements and treatments are stored in wide form across many indexed columns.

## 2.2 Row meaning in the raw dataset

In the original data:

- **1 row = 1 road segment**, typically a 100 m segment

Each row contains:

- segment key fields
- segment metadata
- traffic attributes
- geometric or road class attributes
- treatment history in indexed columns
- pavement measurement history in indexed columns

## 2.3 Important raw column groups

### Segment key columns

These identify the segment:

- `ELY`
- `Tie`
- `Ajorata`
- `Kaista`
- `Aosa`
- `Aet`
- `Losa`
- `Let`

These columns together form the segment key used throughout the transformation.

### Traffic and segment attributes

Common segment-level or road-level attributes include:

- `KVL`
- `KVL_raskas`
- `KVL_kaista`
- `Nopeus`
- `Toim_lk`
- `Pituus`

These are treated as segment attributes and carried into the final event dataset.

### PTM measurement history

Historical pavement measurements are stored in indexed columns such as:

- `PTM_pvm_i` = measurement date
- `Iri_i` = IRI value for measurement slot `i`
- `Ura_max_i` = URA value for measurement slot `i`

Example:

- `PTM_pvm_1`
- `Iri_1`
- `Ura_max_1`
- `PTM_pvm_2`
- `Iri_2`
- `Ura_max_2`
- ...

In the original data, the history is encoded across columns rather than rows.

### TP treatment history

Historical treatment / maintenance events are stored in indexed columns such as:

- `Tp_pvm_i` = treatment date
- `Tp_pinta_i` = treatment surface / pavement-related descriptor
- `Tp_työmen_i` or `Tp_tyomen_i` = treatment method descriptor

Example:

- `Tp_pvm_1`
- `Tp_pinta_1`
- `Tp_työmen_1`
- `Tp_pvm_2`
- `Tp_pinta_2`
- `Tp_työmen_2`
- ...

## 2.4 Structural limitations of the original dataset

The raw dataset is rich, but it is difficult to use directly for machine learning because it is in wide historical format.

The main limitations are:

- time history is spread across many columns instead of rows
- the number of valid historical observations varies by segment
- temporal ordering must be inferred from column indices and dates
- measurements and treatments are stored in separate indexed histories
- treatment events occur between measurements and are not naturally aligned to targets
- feature engineering becomes awkward and brittle in wide format

Because of these limitations, the raw dataset is not a convenient canonical source for downstream modeling.

---

# 3. Why the Dataset Was Restructured

The transformation was designed to solve the main engineering and modeling problems in the raw data.

## 3.1 Main design goals

The redesign aimed to:

- convert history into explicit chronological rows
- preserve both measurements and treatments
- preserve treatment metadata instead of collapsing it away
- keep multiple condition variables available
- support both lifecycle-only and full-history modeling
- support future feature engineering choices not yet decided
- produce a single canonical parquet file instead of many specialized outputs

## 3.2 Why not keep only a measurement panel

A measurement-only panel is convenient for some models, but it loses flexibility if treatment events are reduced to coarse indicators.

For example, if treatment rows are collapsed into only:

- `has_TP_interval`
- `tp_count_interval`

then it becomes impossible to later recover:

- exact treatment dates
- treatment ordering inside an interval
- time since last treatment
- treatment-type-specific counts
- treatment-method-specific effects
- richer event-history sequences

To avoid this information loss, the final design preserves both PTM and TP events explicitly in one chronological event table.

## 3.3 Why use one canonical dataset

Instead of producing many final datasets for different modeling scenarios, the pipeline now produces one canonical dataset that can be used to derive downstream datasets as needed.

This reduces duplication and makes it easier to:

- document one source of truth
- reproduce modeling data consistently
- change feature engineering later without re-reading the raw wide file every time

---

# 4. Concept of the New Dataset

The new dataset is a **unified chronological event-history table**.

## 4.1 Row meaning in the final dataset

In the final dataset:

- **1 row = 1 dated event for 1 road segment**

An event can be one of two types:

- `PTM` = pavement measurement event
- `TP` = treatment / maintenance event

This means the final dataset combines both measurement history and treatment history into one time-ordered event stream.

## 4.2 Event chronology

Within each segment, rows are ordered by:

1. `event_date`
2. `event_order`
3. event-specific slot index

Same-day ordering is handled explicitly:

- `TP` rows come before `PTM` rows on the same date

This allows same-day treatment events to be considered part of the interval leading into the measurement on that date.

## 4.3 What this structure enables

The unified event format makes it possible to:

- reconstruct complete event history for a segment
- build measurement-only modeling datasets later
- build sequence inputs from both measurements and treatments
- define targets at measurement events
- derive arbitrary lookback windows
- use exact treatment timing and type in feature engineering
- restrict models to one lifecycle or use full segment history

---

# 5. Transformation Overview

The transformation from the raw wide file to the final event file is conceptually done in two stages.

## 5.1 Stage 1: Extract wide histories into normalized event tables

The raw parquet file is read in streaming fashion to avoid excessive memory use.

For each row-group and chunk:

- PTM history columns are scanned
- TP history columns are scanned
- valid PTM rows are extracted into a temporary PTM event table
- valid TP rows are extracted into a temporary TP event table
- temporary data is written bucket-by-bucket to disk for memory-safe processing

At this stage, wide histories are converted into long event rows.

### PTM extraction

Each valid measurement slot becomes one row with:

- segment key
- event date
- event type = `PTM`
- `ptm_idx`
- `IRI`
- `URA`
- segment/static attributes

### TP extraction

Each valid treatment slot becomes one row with:

- segment key
- event date
- event type = `TP`
- `tp_idx`
- `Tp_pinta`
- `Tp_tyomen`
- segment/static attributes

## 5.2 Stage 2: Build unified event history and derive context

For each segment bucket:

- PTM rows are ordered chronologically
- TP rows are ordered chronologically
- PTM rows receive transition and lifecycle context
- TP rows receive context from surrounding PTM rows
- PTM and TP rows are concatenated into one event history
- the bucket outputs are finally consolidated into one parquet file

---

# 6. Data Cleaning and Normalization Rules

The transformation includes a number of data cleanup rules inherited from the earlier pipeline design.

These rules are important because they affect what is considered a valid event.

## 6.1 Date cleaning

### TP dates

TP dates are converted to datetime and cleaned as follows:

- placeholder year `1900` is treated as missing
- dates earlier than `1950-01-01` are treated as invalid and removed
- only TP years greater than or equal to 1950 are kept

### PTM dates

PTM dates are converted to datetime and cleaned as follows:

- invalid or unparsable dates become missing
- PTM years earlier than 1950 are removed

## 6.2 Measurement value cleaning

Measurement variables are converted to numeric and clamped to valid ranges.

### IRI

`IRI` is kept only if it falls in the range:

- `0.0 <= IRI <= 20.0`

Values outside this range are replaced with missing.

### URA

`URA` is kept only if it falls in the range:

- `0.0 <= URA <= 80.0`

Values outside this range are replaced with missing.

## 6.3 Deduplication

### PTM rows

PTM rows are deduplicated using:

- segment key
- `event_date`
- `event_type`
- `ptm_idx`

### TP rows

TP rows are deduplicated using:

- segment key
- `event_date`
- `event_type`
- `tp_idx`
- `Tp_pinta` when available
- `Tp_tyomen` when available

The TP deduplication is intentionally less aggressive so that same-date treatment rows with different treatment metadata are not accidentally collapsed away.

---

# 7. Derived Context Added to PTM Rows

PTM rows are the rows where pavement state is directly observed, so most deterioration and lifecycle logic is computed from PTM rows.

## 7.1 Transition variables

For each PTM row, the dataset computes the previous measurement context within the same segment.

### `prev_meas_date`
Date of the previous PTM measurement for the same segment.

### `next_meas_date`
Date of the next PTM measurement for the same segment.

### `prev_URA`
URA at the previous measurement.

### `prev_IRI`
IRI at the previous measurement.

### `delta_URA`
Difference between current and previous URA.

### `delta_IRI`
Difference between current and previous IRI.

### `Delta_t_days`
Days between current and previous measurement.

### `Delta_t_years`
Years between current and previous measurement.

For the first PTM observation of a segment, these previous-state fields are missing.

## 7.2 Treatment count in measurement intervals

For each PTM row, the dataset counts how many TP events occurred in:

- `(previous measurement date, current measurement date]`

This produces:

### `tp_count_interval`
Number of TP events between the previous and current PTM measurement.

### `has_TP_interval`
Boolean flag indicating whether any TP event occurred in the interval.

Because same-day ordering is defined as TP before PTM, a TP on the same date as a PTM measurement is included in the interval ending at that PTM row.

---

# 8. Lifecycle and Reset Logic

The dataset also adds lifecycle-related interpretation to PTM rows.

This logic is based on the earlier modeling pipeline and is preserved as an analytical layer on top of the raw event history.

## 8.1 Major reset detection

A PTM row is marked as a major reset if one of the following is true:

### Known reset

- there was at least one TP event in the interval, and
- `delta_URA <= -1.0`

### Phantom reset

- there was no TP event in the interval, and
- `delta_URA <= -3.0`

This is intended to detect strong pavement-condition resets, including cases where a treatment may be missing from the recorded TP history.

## 8.2 Lifecycle-related columns

### `is_major_reset`
Boolean flag indicating that the PTM row is treated as the start of a new major lifecycle after a reset.

### `is_phantom_reset`
Boolean flag indicating that the reset is inferred from a large URA drop without a recorded TP event.

### `is_minor_treatment`
Boolean flag indicating that a TP occurred in the interval but the event is not classified as a major reset.

### `cycle_num`
Cumulative lifecycle index within the segment.

### `Lifecycle_ID`
Unique lifecycle identifier formed as:

`Segment_ID + "_C" + cycle_num`

### `Pavement_Age_years`
Age in years since the start of the lifecycle.

### `Measurement_Idx`
Index of the measurement within the lifecycle.

### `Initial_URA`
URA value at the first measurement in the lifecycle.

### `Minor_TP_Count`
Cumulative number of minor treatments within the lifecycle.

## 8.3 Important interpretation note

Lifecycle logic is **derived**, not raw.

It is useful for analysis and model construction, but it should be understood as an interpretation based on rules rather than a directly observed ground-truth field from the raw source.

---

# 9. Context Added to TP Rows

TP rows preserve treatment events as first-class rows in the chronology.

These rows do not directly observe pavement condition, so they do not receive PTM-only transition variables such as `delta_URA` or `delta_IRI`.

However, they do receive surrounding measurement context when available.

## 9.1 Surrounding measurement fields on TP rows

### `prev_meas_date`
Most recent PTM measurement date at or before the TP event.

### `next_meas_date`
Next PTM measurement date at or after the TP event.

### `days_since_prev_meas`
Days between the TP event and the previous measurement.

### `days_until_next_meas`
Days between the TP event and the next measurement.

## 9.2 Lifecycle context on TP rows

TP rows also receive lifecycle context from the most recent PTM row at or before the TP date, when such a PTM exists.

This includes:

- `Lifecycle_ID`
- `cycle_num`
- `Initial_URA`
- `Pavement_Age_years`

This allows treatment rows to be placed inside the broader lifecycle chronology even though the lifecycle itself is derived from PTM behavior.

---

# 10. Final Dataset Format

## 10.1 Final output file

The final canonical dataset is stored as a single parquet file.

Example output name:

`road_event_history_v1.parquet`

## 10.2 Row format

Each row is one event for one segment.

Possible event types:

- `PTM`
- `TP`

## 10.3 Ordering

To correctly interpret history, the dataset should be processed in segment-specific chronological order.

The recommended ordering is:

- `Segment_ID`
- `event_date`
- `event_order`
- `ptm_idx` / `tp_idx` as tie-breakers when relevant

## 10.4 Key identifiers

### Raw segment key fields

- `ELY`
- `Tie`
- `Ajorata`
- `Kaista`
- `Aosa`
- `Aet`
- `Losa`
- `Let`

### Derived segment identifier

### `Segment_ID`
String identifier created by concatenating the segment key columns.

### `Lifecycle_ID`
Derived lifecycle identifier for rows where lifecycle context is available.

### `Event_Idx`
Sequential event index within the segment after all events are sorted chronologically.

---

# 11. Key Columns in the Final Dataset

The exact set of columns may depend slightly on the available raw source columns, but the canonical output is designed around the following fields.

## 11.1 Core event fields

### `event_date`
Date of the event.

### `event_type`
Type of event:

- `PTM` for measurement
- `TP` for treatment

### `event_order`
Tie-break order for same-day events.

- `0` for TP
- `1` for PTM

### `ptm_idx`
Original PTM slot index from the raw wide file. Present on PTM rows.

### `tp_idx`
Original TP slot index from the raw wide file. Present on TP rows.

## 11.2 Measurement fields

### `IRI`
Measured IRI value. Usually present on PTM rows only.

### `URA`
Measured URA value. Usually present on PTM rows only.

## 11.3 Treatment fields

### `Tp_pinta`
Treatment surface or related treatment descriptor from the raw TP history.

### `Tp_tyomen`
Treatment method descriptor from the raw TP history.

## 11.4 PTM transition fields

These are mainly meaningful on PTM rows:

- `prev_IRI`
- `prev_URA`
- `delta_IRI`
- `delta_URA`
- `Delta_t_days`
- `Delta_t_years`

## 11.5 Measurement context fields

These can be meaningful for both PTM and TP rows:

- `prev_meas_date`
- `next_meas_date`
- `days_since_prev_meas`
- `days_until_next_meas`

## 11.6 Lifecycle fields

- `Measurement_Idx`
- `cycle_num`
- `Lifecycle_ID`
- `Pavement_Age_years`
- `Initial_URA`
- `Minor_TP_Count`
- `is_major_reset`
- `is_phantom_reset`
- `is_minor_treatment`

## 11.7 Interval treatment summary fields

These are computed relative to PTM measurement intervals:

- `tp_count_interval`
- `has_TP_interval`

## 11.8 Segment/static attributes

Depending on source availability, the final dataset may include:

- `KVL`
- `KVL_raskas`
- `KVL_kaista`
- `Nopeus`
- `Toim_lk`
- `Pituus`

These are repeated across event rows for the segment.

---

# 12. How to Interpret Missing Values

Because the final dataset combines different event types into one table, many columns are intentionally sparse.

This is expected.

## 12.1 Missing values on TP rows

On `TP` rows, the following are usually missing because there is no direct pavement measurement at the treatment event:

- `IRI`
- `URA`
- `prev_IRI`
- `prev_URA`
- `delta_IRI`
- `delta_URA`
- `Delta_t_days`
- `Delta_t_years`
- some lifecycle transition flags

## 12.2 Missing values on first PTM rows

On the first PTM row for a segment, previous-measurement fields are usually missing:

- `prev_meas_date`
- `prev_URA`
- `prev_IRI`
- `delta_URA`
- `delta_IRI`
- `Delta_t_days`
- `Delta_t_years`

This is normal because there is no earlier measurement for comparison.

## 12.3 Missing lifecycle context on early TP rows

If a TP event occurs before the first available PTM measurement for a segment, lifecycle-related fields on that TP row may be missing because lifecycle assignment is anchored to PTM history.

---

# 13. How the New Dataset Differs from the Earlier Panel Design

An earlier design produced a measurement-centric panel where each row represented one PTM measurement and treatment history was reduced to interval summary fields.

The new event-history dataset differs in important ways.

## 13.1 What the earlier panel kept

The earlier panel mainly kept:

- PTM measurement rows
- previous measurement context
- interval treatment counts
- lifecycle segmentation
- selected static features

## 13.2 What the earlier panel lost

The earlier panel dropped or compressed:

- explicit TP rows
- treatment dates as first-class events
- `Tp_pinta`
- `Tp_tyomen`
- exact treatment timing inside intervals
- richer full-history sequencing options

## 13.3 What the new event-history dataset preserves

The new canonical dataset preserves:

- PTM rows
- TP rows
- treatment metadata
- measurement variables `URA` and `IRI`
- treatment chronology
- segment chronology across lifecycles
- lifecycle annotations as derived context

This makes it a better long-term canonical source.

---

# 14. Recommended Ways to Use the Dataset

The dataset is designed to be a canonical source, not necessarily the final model matrix.

In most projects, you should derive task-specific training datasets from this file.

## 14.1 General rule

When building features or targets, first sort the data by:

- `Segment_ID`
- `event_date`
- `event_order`
- slot index where needed

Then decide what rows are used as:

- input history
- prediction targets
- filtering boundaries such as lifecycle restrictions

## 14.2 Predicting condition at the next measurement

A common task is:

- predict `URA` or `IRI` at the next PTM event

Recommended approach:

1. filter target rows to `event_type == "PTM"`
2. build input history from earlier rows of the same segment
3. choose whether history includes:
   - only earlier PTM rows
   - PTM + TP rows
4. generate target by shifting PTM rows forward within the segment or within the lifecycle

## 14.3 Predicting several measurements ahead

For tasks like predicting 3 measurements ahead:

1. restrict to PTM rows as candidate targets
2. define the target as the PTM value at horizon `k` within the same segment or lifecycle
3. build features from all earlier events available before the prediction time

Because the dataset is event-based, you can choose whether to use:

- only PTM history
- only recent history
- full event history including TP rows

## 14.4 Lifecycle-only modeling

If the goal is to model deterioration inside one lifecycle:

- group by `Lifecycle_ID`
- use only rows within the lifecycle
- optionally exclude TP rows if the model is measurement-only
- or keep TP rows if treatment events are part of the sequence input

## 14.5 Whole-history modeling

If the goal is to use the full road history:

- group by `Segment_ID`
- use all prior events before the target PTM row
- let lifecycle boundaries be features rather than hard data cuts

This is one of the main reasons the unified event-history structure was chosen.

---

# 15. Example Interpretations

## 15.1 Example PTM row

A PTM row might be interpreted as:

- a measured pavement state for one road segment
- with previous measured condition available
- with time since previous measurement available
- with number of treatment events since previous measurement available
- with lifecycle identity and pavement age available

This row is usually a natural candidate target row for supervised learning.

## 15.2 Example TP row

A TP row might be interpreted as:

- a maintenance or restoration event
- with treatment descriptors preserved
- located between two measurements
- placed into the segment chronology
- optionally associated with a lifecycle

This row is usually not a direct prediction target for pavement-condition forecasting, but it is valuable as context in the input history.

---

# 16. Engineering Notes and Practical Guidance

## 16.1 The dataset is canonical, not task-final

Do not assume this file is the final matrix for every model.

Instead, treat it as a canonical event-history source from which you derive:

- model-specific windows
- lag features
- event encodings
- target definitions
- lifecycle-filtered subsets

## 16.2 Preserve ordering in all downstream steps

Many features depend on chronology.

If rows are not sorted correctly within each segment, the meaning of:

- previous measurement fields
- interval treatment counts
- lifecycle ordering
- sequence inputs

can break.

## 16.3 Be explicit about target definition

When building a model, define clearly:

- which rows are prediction targets
- whether targets are only PTM rows
- whether horizon is measured in:
  - number of measurements ahead, or
  - calendar time ahead

The dataset supports both, but they are different tasks.

## 16.4 Use event type consciously

In downstream modeling, decide explicitly how to encode event type.

Possible approaches include:

- use only PTM rows
- use PTM rows and summarize TP rows into interval features
- use both PTM and TP rows as a sequence
- convert TP rows into event tokens or categorical treatment features

The dataset is designed to keep all of these options open.

## 16.5 Treat lifecycle labels as derived features

`Lifecycle_ID`, `cycle_num`, and reset flags are useful, but they come from heuristic rules.

They should be treated as derived analytical context, not unquestioned truth.

---

# 17. Known Assumptions and Caveats

The engineer using this dataset should be aware of the following assumptions.

## 17.1 Reset logic is heuristic

Major reset detection is based on URA-drop rules and treatment occurrence rules.

This is useful but not perfect. Some real resets may be missed, and some inferred resets may not reflect actual major rehabilitation.

## 17.2 Some static attributes may be time-varying in reality

Fields such as traffic attributes may change over calendar time in the real world, but in the current transformation they are carried from the segment row into event rows as available in the raw unified file.

That means they should be interpreted as source-provided attributes attached to the segment, not necessarily as a fully time-resolved historical record.

## 17.3 Source data quality still matters

The transformation cleans obvious invalid dates and out-of-range measurements, but it does not guarantee that all remaining source values are semantically correct.

Unexpected patterns should still be investigated against the raw data when needed.

## 17.4 Sparse fields are normal

Because this is a mixed event table, it is normal for many columns to be meaningful only for one event type.

This is not a defect; it is part of the design.

---

# 18. Suggested Workflow for New Engineers

A practical onboarding workflow for using this dataset is:

1. Load the parquet file.
2. Inspect the schema and distinct `event_type` values.
3. Sort rows by `Segment_ID`, `event_date`, `event_order`.
4. Pick a few example segments and inspect their full chronology manually.
5. Verify how PTM and TP rows interleave.
6. Decide what the modeling target is.
7. Derive a task-specific training view from the canonical event history.

A good first manual check is to inspect one segment and confirm that:

- PTM dates are in the expected order
- TP events appear in the right place between measurements
- same-day TP rows appear before same-day PTM rows
- lifecycle changes happen where large resets are expected

---

# 19. Summary

The original dataset stored road-segment history in wide form, which made it awkward to use for machine learning and temporal analysis.

The new dataset restructures that information into a single chronological event-history table where each row is either:

- a PTM measurement event, or
- a TP treatment event

The transformation preserves:

- segment keys
- treatment metadata
- condition measurements
- event chronology
- lifecycle-related derived context

The resulting dataset is intended to be the **canonical source** for downstream modeling and feature engineering.

It is especially useful because it supports both:

- simple measurement-based supervised learning workflows, and
- richer sequence or event-history-based modeling workflows

The most important idea for users of this dataset is:

> this file is not just a table of measurements; it is a time-ordered history of both condition observations and treatment actions for each road segment.

That is what gives it flexibility for future model design.

---

# 20. Minimal Column Cheat Sheet

This section gives a compact reference for the most important columns.

| Column                                                           | Meaning                                    |
| ---------------------------------------------------------------- | ------------------------------------------ |
| `Segment_ID`                                                     | Derived unique segment identifier          |
| `event_date`                                                     | Date of event                              |
| `event_type`                                                     | `PTM` or `TP`                              |
| `event_order`                                                    | Same-day ordering, TP before PTM           |
| `ptm_idx`                                                        | Original PTM slot index                    |
| `tp_idx`                                                         | Original TP slot index                     |
| `URA`                                                            | Measured rutting value on PTM rows         |
| `IRI`                                                            | Measured roughness value on PTM rows       |
| `Tp_pinta`                                                       | Treatment descriptor                       |
| `Tp_tyomen`                                                      | Treatment method descriptor                |
| `prev_URA`                                                       | Previous PTM URA value                     |
| `prev_IRI`                                                       | Previous PTM IRI value                     |
| `delta_URA`                                                      | Change in URA from previous PTM            |
| `delta_IRI`                                                      | Change in IRI from previous PTM            |
| `Delta_t_days`                                                   | Days since previous PTM                    |
| `tp_count_interval`                                              | Number of TP events since previous PTM     |
| `has_TP_interval`                                                | Whether any TP occurred since previous PTM |
| `is_major_reset`                                                 | Heuristic flag for major reset             |
| `is_phantom_reset`                                               | Heuristic reset without recorded TP        |
| `is_minor_treatment`                                             | TP interval without major reset            |
| `cycle_num`                                                      | Lifecycle index within segment             |
| `Lifecycle_ID`                                                   | Derived lifecycle identifier               |
| `Measurement_Idx`                                                | PTM index within lifecycle                 |
| `Pavement_Age_years`                                             | Years since lifecycle start                |
| `KVL`, `KVL_raskas`, `KVL_kaista`, `Nopeus`, `Toim_lk`, `Pituus` | Segment/static attributes                  |

---
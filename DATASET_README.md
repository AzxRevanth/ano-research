# AMPds2 Feature Engineering & Anomaly Injection Pipeline (`main.ipynb`)

## 1. Overview & Purpose

The `main.ipynb` notebook implements an end-to-end data processing, synthetic anomaly injection, and automated feature engineering pipeline for smart meter anomaly detection and diagnosis.

Starting from raw **1-minute electricity sub-metering readings** and **hourly meteorological observations** from the **AMPds2 (Almanac of Minutely Power dataset)**, this pipeline transforms raw continuous time series into an aligned, high-dimensional **15-minute sliding window feature matrix**:
- **Dataset File:** `pipeline_cache/ampds_behavior_context_labeled_features.csv`
- **Observations:** **5,856** fifteen-minute windows (September 1, 2012 – October 31, 2012)
- **Features:** **830** total predictor features (817 FRESH-pruned behavior features + 13 context features)
- **Labels:** Binary label (`is_anomaly`) and granular multi-class label (`anomaly_type` across **14 distinct anomaly categories**)
- **Anomaly Prevalence:** **758** anomalous windows (12.94% contamination rate) and **5,098** normal windows

---

## 2. Source Data Ingestion

The pipeline ingests three raw CSV datasets located in the root repository directory:

| Dataset File | Sampling Rate | Description | Key Variables Extracted |
| :--- | :---: | :--- | :--- |
| **`Electricity_WHE.csv`** | 1 minute | Whole-house main electrical service (240V split-phase mains). | Voltage ($V$), Current ($I$), Active Real Power ($P$ in W), Reactive Power ($Q$ in VAR), Apparent Power ($S$ in VA). |
| **`Electricity_P.csv`** | 1 minute | Sub-metered branch circuits monitoring individual appliance active power. | `FRE` (Fridge/Freezer), `HPE` (Heat Pump), `DWE` (Dishwasher), `CWE` (Clothes Washer/Dryer), `WOE` (Wall Oven), `B1E` (Bedroom/Basement Plug Loads). |
| **`Climate_HourlyWeather.csv`** | 1 hour | Historical weather observations from Environment Canada (Vancouver Intl. Airport station). | `Temp (C)`, `Rel Hum (%)`, `Wind Spd (km/h)`, `Visibility (km)`, `Stn Press (kPa)`, text `Weather` description. |

---

## 3. Data Ingestion, Cleaning & Temporal Alignment

1. **Timestamp Normalization & Timezone Localization:**
   - UNIX epoch timestamps (`UNIX_TS` / `unix_ts`) in `Electricity_WHE.csv` and `Electricity_P.csv` are parsed to UTC datetime objects and localized to Vancouver local time (`America/Vancouver`).
   - Mixed-format date strings in `Climate_HourlyWeather.csv` are parsed and localized to `America/Vancouver`, with Daylight Saving Time (DST) edge cases resolved (`ambiguous='NaT', nonexistent='shift_forward'`).
   - Low-information data quality flags (`Data Quality`, `Temp Flag`, `Hmdx`, `Wind Chill`, etc.) are stripped.

2. **Temporal Scoping:**
   - The time series is filtered to a continuous 2-month observation window:
     $$\text{Start: } \mathbf{2012\text{-}09\text{-}01\text{ 00:00:00}} \quad \longrightarrow \quad \text{End: } \mathbf{2012\text{-}10\text{-}31\text{ 23:59:00}}$$
   - Total duration: **61 days**, corresponding to **87,840 continuous 1-minute measurements**.

3. **Sensor Merging & Imputation:**
   - An inner join combines whole-house electrical variables (`V, I, P, Q, S`) with the 6 appliance circuits (`FRE, HPE, DWE, CWE, WOE, B1E`), producing an 11-channel behavioral stream:
     $$\mathbf{X}_{\text{behavior}} \in \mathbb{R}^{87840 \times 11}$$
   - Any sporadic missing readings are imputed using forward-fill (`ffill()`) followed by backward-fill (`bfill()`).

4. **Weather Regime Encoding & Downsampling Alignment:**
   - Raw weather description strings are categorized into 7 standardized weather regimes:
     $$\text{Regimes} = \{\text{Clear}, \text{Cloudy}, \text{Rain}, \text{Fog}, \text{Snow}, \text{Thunderstorm}, \text{Other}\}$$
   - Regimes are one-hot encoded into binary dummy features (`wx_Clear`, `wx_Cloudy`, `wx_Rain`, `wx_Fog`, etc.).
   - The hourly climate indicators and binary weather dummies are merged into the 1-minute electricity table and forward-filled down to minutely resolution.

---

## 4. Synthetic Anomaly Injection Framework

Real-world smart meter datasets rarely contain rich ground-truth labeled anomaly varieties with minute-level precision. To evaluate both anomaly detection and fine-grained root-cause diagnosis, `main.ipynb` implements a synthetic anomaly injection framework across **14 distinct anomaly types** spanning **4 structural anomaly families**:

```
                       ANOMALY TAXONOMY (14 TYPES)
                                    │
    ┌─────────────────┬─────────────┴───────────┬─────────────────┐
    ▼                 ▼                         ▼                 ▼
Contextual        Sequential & Drift        Collective        Point / Transient
(5 Types)         (5 Types)                 (2 Types)         (2 Types)
- Heating Warm    - Stuck On                - Multi-High      - Power Spike
- Unusual Hours   - Stuck Off               - Impossible      - Sensor Glitch
- Low Occupancy   - Drift Increase            Combo
- Weekend/Weekday - Drift Decrease
- Weekday/Weekend - Sustained Overload
```

### Detailed Breakdown of Injected Anomaly Types

| Family | Anomaly Type (`anomaly_type`) | Injection Logic & Physics | Window Count |
| :--- | :--- | :--- | :---: |
| **Contextual** | `heating_on_warm_day` | Heat pump active power (`HPE`) boosted by $3\times \text{–} 5\times \max(\text{HPE}) + 500\text{W}$ during warm, sunny days ($\text{Temp} > 15^\circ\text{C}$, $\text{wx\_Clear}=1$). | 63 |
| **Contextual** | `appliance_unusual_hours` | Wall oven (`WOE`) activated at $1.5\times \text{–} 3\times$ its 90th percentile load in the middle of the night (2:00 AM – 5:00 AM). | 78 |
| **Contextual** | `high_usage_low_occupancy` | Total household active power ($P$) elevated by $+700\text{W} \text{–} 1400\text{W}$ during unoccupied weekday hours (10:00 AM – 3:00 PM). | 77 |
| **Contextual** | `weekend_pattern_on_weekday` | Clothes washer/dryer (`CWE`) running on a weekday at $1.5\times \text{–} 2.5\times$ the historical weekend average. | 63 |
| **Contextual** | `weekday_pattern_on_weekend` | Clothes washer/dryer (`CWE`) running on a weekend at $1.5\times \text{–} 2.5\times$ the historical weekday average. | 61 |
| **Sequential** | `stuck_appliance_on` | Dishwasher (`DWE`) clamped at its 75th percentile consumption for a continuous 90-minute duration. | 89 |
| **Sequential** | `stuck_appliance_off` | Heat pump (`HPE`) unexpectedly clamped to $0.0\text{W}$ continuously for 90 minutes. | 88 |
| **Sequential** | `gradual_drift_increase` | Bedroom/basement load (`B1E`) slowly ramps upward by $+80\% \text{–} 150\%$ over a 60-minute window. | 40 |
| **Sequential** | `gradual_drift_decrease` | Bedroom/basement load (`B1E`) slowly decays downward over a 60-minute window. | 23 |
| **Sequential** | `sustained_overload` | Total active power ($P$) elevated by $+1000\text{W} \text{–} 1800\text{W}$ above normal for 60 consecutive minutes. | 53 |
| **Collective** | `multiple_high_power_simultaneous` | Heat pump (`HPE`), Dishwasher (`DWE`), and Dryer (`CWE`) all simultaneously draw $1.2\times \text{–} 1.8\times$ their 90th percentiles. | 34 |
| **Collective** | `impossible_appliance_combo` | Heat pump (`HPE`) and Wall oven (`WOE`) fire concurrently at $1.3\times \text{–} 2.0\times$ their 95th percentile draw. | 31 |
| **Point** | `power_spike` | Sudden impulsive surge in total active power ($P$) by $+900\text{W} \text{–} 1700\text{W}$ within a single 15-minute window. | 34 |
| **Point** | `sensor_glitch` | Voltage ($V$) scaled by $0.5\times \text{–} 1.8\times$ and current ($I$) scaled by $0.3\times \text{–} 2.5\times$, violating physical power triangle constraints ($P \approx V \cdot I$). | 24 |

*All injected anomaly metadata and timestamps are persisted to `pipeline_cache/injected_anomaly_labels.csv`.*

---

## 5. Sliding-Window Rolling (15-Minute Windows)

1. **Multivariate Long Format Conversion:**
   The wide merged table is melted into TSFresh long format:
   ```python
   long_df = merged.reset_index().melt(
       id_vars=['timestamp'],
       value_vars=BEHAVIOR_COLS,
       var_name='sensor',
       value_name='value'
   )
   long_df['entity_id'] = 'house'
   ```
   > **Design Note:** Assigning a constant `entity_id = 'house'` is critical. It forces TSFresh to roll all 11 multivariate sensor streams together within the same time window, rather than rolling each sensor independently.

2. **Window Rolling:**
   Using `tsfresh.utilities.dataframe_functions.roll_time_series`:
   - `max_timeshift = 14`, `min_timeshift = 14` (exact 15-minute window span)
   - `rolling_direction = 15` (15-minute stride, ensuring non-overlapping windows)
   - Total sliding windows generated:
     $$N_{\text{windows}} = \frac{87840 \text{ minutes}}{15 \text{ min/window}} = \mathbf{5,856 \text{ windows}}$$
   - Cached to disk as: `pipeline_cache/df_rolled.parquet`.

---

## 6. TSFresh Feature Extraction & FRESH Pruning

1. **Multivariate Feature Extraction:**
   Using `tsfresh.feature_extraction.extract_features` with `EfficientFCParameters()` distributed across 10 CPU workers (`n_jobs=10`, `chunksize=500`):
   - Computes statistical, spectral, temporal, and non-linear characteristics across all 11 sensor streams (e.g., Fourier transform coefficients, energy ratios, autoregressive parameters, approximate entropy, peak counts, quantiles, variance, skewness, kurtosis).
   - Missing and infinite values are imputed using `tsfresh.utilities.dataframe_functions.impute()`.
   - The tuple-string index `(entity_id, timestamp)` is parsed into a clean datetime index `window_id`.
   - Cached to disk as: `pipeline_cache/raw_features.parquet`.

2. **Supervised FRESH Feature Selection (Pruning):**
   Using `tsfresh.select_features(raw_features, true_target)`:
   - Each extracted feature undergoes rigorous hypothesis testing against the ground-truth anomaly label using the **Benjamini-Yekutieli procedure** to control False Discovery Rate (FDR).
   - Prunes thousands of raw features down to **817 statistically significant behavior features** strongly correlated with anomalous transitions.

---

## 7. Context Vector Construction & Final Dataset Assembly

1. **Context Vector Engineering:**
   The climate measurements and temporal indicators are aggregated over each 15-minute window:
   - **Continuous Weather Metrics:** Mean temperature, humidity, wind speed, visibility, and atmospheric pressure.
   - **Weather Regimes:** Max-pooled dummy indicators (`wx_Clear`, `wx_Cloudy`, `wx_Rain`, `wx_Fog`).
   - **Cyclic Temporal Encodings:** Transforms hour of the day and day of the week into continuous trigonometric coordinates to preserve cyclical continuity (preventing artificial discontinuity between 23:59 and 00:00, or Sunday and Monday):
     $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
     $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
   - **Weekend Indicator:** Binary flag `is_weekend` (1 for Saturday/Sunday, 0 for weekdays).

2. **Final Concatenation:**
   An inner join combines the 817 FRESH-pruned behavior features with the 13 context features, appending `is_anomaly` and `anomaly_type`:
   ```python
   combined = behavior_vector.join(context, how='inner').dropna()
   combined['is_anomaly'] = combined.index.isin(anomaly_df['window_id']).astype(int)
   combined['anomaly_type'] = combined.index.map(label_lookup).fillna('normal')
   ```

3. **Output File:**
   - Saved to: `pipeline_cache/ampds_behavior_context_labeled_features.csv`
   - **Final Shape:** **5,856 rows $\times$ 833 columns** (Index: `window_id`, 830 predictor features, 2 label columns, plus metadata).

---

## 8. Dataset Summary & Benchmark Split Composition

### Class Distribution
- **Normal Windows (`is_anomaly = 0`):** 5,098 (87.06%)
- **Anomalous Windows (`is_anomaly = 1`):** 758 (12.94%) across 14 anomaly types.

### Stratified 70 / 15 / 15 Benchmark Splits
As utilized across all modeling experiments (`best_model_yet.py`, `complete_system_1.py`, `complete_system_2.py`), the dataset is partitioned using stratified sampling on `anomaly_type` (`random_state=42`):

| Dataset Split | Normal Windows | Anomalous Windows | Total Windows | Contamination Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Train (70%)** | 3,568 | 530 | 4,098 | 12.93% |
| **Validation (15%)** | 765 | 114 | 879 | 12.97% |
| **Test (15%)** | 765 | 114 | 879 | 12.97% |
| **Full Dataset (100%)** | **5,098** | **758** | **5,856** | **12.94%** |

---

## 9. Caching Architecture & Reproducibility

To avoid recomputing computationally intensive stages on every run, the pipeline implements disk caching in `pipeline_cache/`:

| Cache File | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| **`df_rolled.parquet`** | Parquet | ~10 MB | 15-minute sliding windows across the multivariate stream. |
| **`raw_features.parquet`** | Parquet | ~75 MB | Imputed high-dimensional TSFresh features prior to pruning. |
| **`injected_anomaly_labels.csv`** | CSV | ~25 KB | Ground-truth log of injected window timestamps and anomaly classes. |
| **`ampds_behavior_context_labeled_features.csv`** | CSV | ~48 MB | **Final complete dataset** used by all detection and clustering pipelines. |

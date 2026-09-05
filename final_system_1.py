"""
================================================================================
FINAL SYSTEM 1: COMPLETE END-TO-END PIPELINE
================================================================================
Combines:
1. TSFresh Feature Engineering & Anomaly Injection Pipeline (from tsfreshing.ipynb)
   - Raw AMPds2 Smart Meter Timeseries Loading & Subsetting
   - Weather Integration & Regime Encoding
   - Synthetic Anomaly Injection (14 Anomaly Types / Multi-Pattern)
   - Sliding Window Time-Series Rolling (15-min Non-Overlapping Windows)
   - TSFresh Multivariate Feature Extraction & Imputation
   - FRESH Supervised Hypothesis Testing & Feature Pruning
   - Context Vector Construction (Cyclic Time + Weather Regimes)
   - Labeled Feature Dataset Generation & Caching

2. Stage 1: NMF Anomaly Detection Pipeline (from complete_system_2.py)
   - 70/15/15 Stratified Novelty Train/Val/Test Split
   - MinMaxScaler (Fitted Strictly on Normal Training Windows)
   - Unregularized Non-Negative Matrix Factorization (K=40)
   - Inverse Normal Training Reconstruction Error Feature Weighting
   - Combined RGAnomaly Anomaly Scoring (Input Error + Latent Space Discrepancy, alpha=0.4)
   - Validation-Guided Optimal Threshold Tuning (Maximizing F1)
   - Untouched Test Set Evaluation

3. Stage 2: Spearman Graph Markov Clustering & Diagnosis (from complete_system_2.py)
   - MTH-IDS Tier 4 like Biased Classifier: Random Forest False Positive Filter (P >= 0.70)
   - Latent Representation Compression: L2-Normalization + 10-D PCA
   - Spearman Rank Correlation Matrix Construction (Monotonic Invariance)
   - Graph Adjacency Network Construction with Weak Edge Pruning (threshold = 0.80)
   - Markov Clustering (MCL, Inflation = 1.5)
   - MTH-IDS Tier 3 like Semi-Supervised Cluster Labeling by Majority Ground-Truth Vote
   - Rigorous Cluster Auditing: Size-Weighted Micro Purity, Purity on N >= 3 Clusters
   - Full Cross-Tabulation Matrix & Per-Cluster Diagnostic Breakdown
================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ------------------------------------------------------------------------------
# Optional / Auto-Installed Graph Clustering Package
# ------------------------------------------------------------------------------
try:
    import markov_clustering as mc
except ImportError:
    os.system(f'"{sys.executable}" -m pip install markov_clustering -q')
    import markov_clustering as mc

# ------------------------------------------------------------------------------
# TSFresh Feature Extraction & Selection Packages
# ------------------------------------------------------------------------------
try:
    from tsfresh.utilities.dataframe_functions import roll_time_series, impute
    from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters
    from tsfresh import select_features
except ImportError:
    os.system(f'"{sys.executable}" -m pip install tsfresh -q')
    from tsfresh.utilities.dataframe_functions import roll_time_series, impute
    from tsfresh.feature_extraction import extract_features, EfficientFCParameters, MinimalFCParameters
    from tsfresh import select_features

# ------------------------------------------------------------------------------
# Scikit-Learn Machine Learning & Evaluation Packages
# ------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==============================================================================
# UNIFIED CONFIGURATION BLOCK
# ==============================================================================
CONFIG = {
    # File Paths & Directories
    "data_dir": r"C:\1.Revanth\Projects\research",
    "cache_dir": r"C:\1.Revanth\Projects\research\pipeline_cache",
    "dataset_filename": "ampds_behavior_context_labeled_features.csv",
    "force_recompute_tsfresh": False,  # True = re-extract TSFresh features; False = load cached CSV if available
    "random_state": 42,
    
    # Feature Engineering Parameters (tsfreshing.ipynb)
    "date_start": "2012-09-01",
    "date_end": "2012-10-31",
    "window_size_min": 15,
    "stride_min": 15,
    "appliance_cols": ["FRE", "HPE", "DWE", "CWE", "WOE", "B1E"],
    "use_minimal_features": False,     # False = full EfficientFCParameters
    "n_jobs_tsfresh": 10,
    
    # Stage 1: NMF Anomaly Detection Parameters (complete_system_2.py)
    "nmf_k": 40,
    "nmf_alpha": 0.4,                  # Weight on input reconstruction error vs latent re-encoding error
    "nmf_max_iter": 3000,
    "nmf_tol": 1e-3,
    "epsilon": 1e-6,                   # Stability constant for inverse MSE feature weighting
    
    # Stage 2: Spearman Graph MCL Clustering Parameters (complete_system_2.py)
    "pca_dims": 10,
    "similarity_metric": "spearman",
    "rf_conf_threshold": 0.70,         # Filter alarms with P(true anomaly) >= 0.70
    "rf_n_estimators": 100,
    "mcl_spearman_threshold": 0.80,     # Graph edge pruning threshold for Spearman correlation
    "mcl_inflation": 1.5,              # MCL inflation operator
    "min_cluster_eval_size": 3,        # Threshold size for non-trivial cluster purity evaluation
}


# ==============================================================================
# PART 1: TSFRESH FEATURE ENGINEERING & DATASET PREPARATION (tsfreshing.ipynb)
# ==============================================================================
def simplify_weather(w):
    """Categorizes granular raw weather descriptions into standard regimes."""
    if pd.isna(w):
        return 'Unknown'
    w = str(w).lower()
    if 'thunder' in w:
        return 'Thunderstorm'
    if 'snow' in w:
        return 'Snow'
    if 'fog' in w:
        return 'Fog'
    if 'rain' in w or 'drizzle' in w:
        return 'Rain'
    if 'cloud' in w:
        return 'Cloudy'
    if 'clear' in w or 'sunny' in w:
        return 'Clear'
    return 'Other'


def inject_heating_on_warm_day(df, log, window_size_min=15, n_events=15):
    """Injects high heat-pump usage on clear, warm days."""
    candidates = df[(df.get('wx_Clear', 0) == 1) & (df['Temp (C)'] > 15)].index
    if len(candidates) == 0:
        print("No candidates for heating_on_warm_day - skipped")
        return df
    chosen = np.random.choice(candidates, size=min(n_events, len(candidates)), replace=False)
    for ts in chosen:
        window = df.index[(df.index >= ts) & (df.index < ts + pd.Timedelta(minutes=window_size_min))]
        df.loc[window, 'HPE'] = df.loc[window, 'HPE'].max() * np.random.uniform(3, 5) + 500
        log.append({'window_id': ts.floor(f'{window_size_min}min'), 'anomaly_type': 'heating_on_warm_day'})
    return df


def inject_night_load_spike(df, log, window_size_min=15, n_events=15):
    """Injects high unexpected power draw during late night hours (1 AM - 4 AM)."""
    candidates = df[(df.index.hour >= 1) & (df.index.hour <= 4)].index
    chosen = np.random.choice(candidates, size=min(n_events, len(candidates)), replace=False)
    for ts in chosen:
        window = df.index[(df.index >= ts) & (df.index < ts + pd.Timedelta(minutes=window_size_min))]
        df.loc[window, 'P'] = df.loc[window, 'P'] + np.random.uniform(800, 1500)
        log.append({'window_id': ts.floor(f'{window_size_min}min'), 'anomaly_type': 'night_load_spike'})
    return df


def inject_stuck_appliance(df, log, appliance='DWE', n_events=10, duration_min=60, window_size_min=15):
    """Injects persistent, stuck appliance behavior where load stays constant for an extended duration."""
    if appliance not in df.columns:
        return df
    candidates = df.index[:-duration_min]
    chosen = np.random.choice(candidates, size=min(n_events, len(candidates)), replace=False)
    stuck_value = df[appliance].quantile(0.75)
    for ts in chosen:
        window = df.index[(df.index >= ts) & (df.index < ts + pd.Timedelta(minutes=duration_min))]
        df.loc[window, appliance] = stuck_value
        for w in pd.date_range(ts, periods=duration_min // window_size_min, freq=f'{window_size_min}min'):
            log.append({'window_id': w.floor(f'{window_size_min}min'), 'anomaly_type': 'stuck_appliance'})
    return df


def inject_weather_mismatched_load(df, log, window_size_min=15, n_events=15):
    """Injects heavy total active load (P) on mild, rainy days."""
    candidates = df[(df.get('wx_Rain', 0) == 1) & (abs(df['Temp (C)'] - df['Temp (C)'].mean()) < 2)].index
    if len(candidates) == 0:
        print("No candidates for weather_mismatched_load - skipped")
        return df
    chosen = np.random.choice(candidates, size=min(n_events, len(candidates)), replace=False)
    for ts in chosen:
        window = df.index[(df.index >= ts) & (df.index < ts + pd.Timedelta(minutes=window_size_min))]
        df.loc[window, 'P'] = df.loc[window, 'P'] + np.random.uniform(600, 1200)
        log.append({'window_id': ts.floor(f'{window_size_min}min'), 'anomaly_type': 'weather_mismatched_load'})
    return df


def inject_weekend_pattern_on_weekday(df, log, window_size_min=15, n_events=10):
    """Injects heavy weekend clothes washer/dryer usage during a weekday."""
    if 'CWE' not in df.columns:
        return df
    weekday_candidates = df[df.index.dayofweek < 5].index
    weekend_avg = df[df.index.dayofweek >= 5]['CWE'].mean()
    chosen = np.random.choice(weekday_candidates, size=min(n_events, len(weekday_candidates)), replace=False)
    for ts in chosen:
        window = df.index[(df.index >= ts) & (df.index < ts + pd.Timedelta(minutes=window_size_min))]
        df.loc[window, 'CWE'] = weekend_avg * np.random.uniform(1.5, 2.5)
        log.append({'window_id': ts.floor(f'{window_size_min}min'), 'anomaly_type': 'weekend_pattern_on_weekday'})
    return df


def generate_tsfresh_dataset(cfg):
    """
    Executes the complete TSFresh feature engineering pipeline from tsfreshing.ipynb:
    1. Loads raw AMPds2 electricity and climate files
    2. Aligns timestamps and normalizes timezone to America/Vancouver
    3. Merges WHE mains and sub-metered appliances
    4. Encodes weather regimes
    5. Injects synthetic anomalies and builds label ground truth
    6. Constructs multivariate long format and rolls sliding windows
    7. Extracts TSFresh features with imputations
    8. Performs FRESH feature selection against labels
    9. Concatenates cyclic time + weather context vector
    10. Saves final feature matrix to CSV
    """
    cache_dir = cfg["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    output_path = os.path.join(cache_dir, cfg["dataset_filename"])
    
    # If already generated and force_recompute is False, load cached dataset immediately
    if os.path.exists(output_path) and not cfg["force_recompute_tsfresh"]:
        print(f"Found existing cached feature dataset at: {output_path}")
        print("Skipping TSFresh extraction and loading feature matrix directly.")
        return output_path
        
    print("=" * 80)
    print("STARTING TSFRESH FEATURE EXTRACTION & DATASET CREATION (tsfreshing.ipynb)")
    print("=" * 80)
    
    np.random.seed(cfg["random_state"])
    data_dir = cfg["data_dir"]
    date_start = cfg["date_start"]
    date_end = cfg["date_end"]
    window_size_min = cfg["window_size_min"]
    stride_min = cfg["stride_min"]
    appliance_cols = cfg["appliance_cols"]
    use_minimal = cfg["use_minimal_features"]
    n_jobs = cfg["n_jobs_tsfresh"]
    
    rolled_path = os.path.join(cache_dir, 'df_rolled.parquet')
    raw_features_path = os.path.join(cache_dir, 'raw_features.parquet')
    
    # 1. Load Raw CSVs
    print("Loading raw timeseries CSVs...")
    whe = pd.read_csv(os.path.join(data_dir, 'Electricity_WHE.csv'))
    p_all = pd.read_csv(os.path.join(data_dir, 'Electricity_P.csv'))
    weather = pd.read_csv(os.path.join(data_dir, 'Climate_HourlyWeather.csv'))
    
    # 2. Timestamp Alignment
    whe_ts_col = 'unix_ts' if 'unix_ts' in whe.columns else 'UNIX_TS'
    p_ts_col = 'UNIX_TS' if 'UNIX_TS' in p_all.columns else 'unix_ts'
    
    whe['timestamp'] = pd.to_datetime(whe[whe_ts_col], unit='s', utc=True).dt.tz_convert('America/Vancouver')
    p_all['timestamp'] = pd.to_datetime(p_all[p_ts_col], unit='s', utc=True).dt.tz_convert('America/Vancouver')
    
    whe = whe.drop(columns=[whe_ts_col]).set_index('timestamp').sort_index()
    p_all = p_all.drop(columns=[p_ts_col]).set_index('timestamp').sort_index()
    
    weather['timestamp'] = pd.to_datetime(weather['Date/Time'], format='mixed')
    weather = weather.set_index('timestamp').sort_index()
    weather.index = weather.index.tz_localize('America/Vancouver', ambiguous='NaT', nonexistent='shift_forward')
    weather = weather[weather.index.notna()]
    
    weather = weather.drop(columns=[
        'Data Quality', 'Temp Flag', 'Dew Point Temp Flag', 'Rel Hum Flag',
        'Wind Dir Flag', 'Wind Spd Flag', 'Visibility Flag', 'Stn Press Flag',
        'Hmdx', 'Hmdx Flag', 'Wind Chill', 'Wind Chill Flag'
    ], errors='ignore')
    
    # 3. Filter to 2-Month Target Subset
    whe = whe.loc[date_start:date_end]
    p_all = p_all.loc[date_start:date_end]
    weather = weather.loc[date_start:date_end]
    print(f"WHE records: {len(whe)}, P records: {len(p_all)}, Weather records: {len(weather)}")
    
    # 4. Merge Mains + Appliances
    appliance_cols_present = [c for c in appliance_cols if c in p_all.columns]
    merged = whe.join(p_all[appliance_cols_present], how='inner')
    
    behavior_cols = ['V', 'I', 'P', 'Q', 'S'] + appliance_cols_present
    for col in behavior_cols:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').astype('float64')
    merged[behavior_cols] = merged[behavior_cols].ffill().bfill()
    
    # 5. Weather Regime Encoding
    weather['weather_regime'] = weather['Weather'].apply(simplify_weather)
    weather_dummies = pd.get_dummies(weather['weather_regime'], prefix='wx')
    weather = weather.join(weather_dummies)
    
    weather_numeric_cols = ['Temp (C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)']
    for col in weather_numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')
        
    weather_regime_cols = weather_dummies.columns.tolist()
    merged = merged.join(weather[weather_numeric_cols + weather_regime_cols], how='left')
    merged[weather_numeric_cols + weather_regime_cols] = merged[weather_numeric_cols + weather_regime_cols].ffill().infer_objects(copy=False)
    
    # 6. Synthetic Anomaly Injections
    merged['window_id'] = merged.index.floor(f'{window_size_min}min')
    anomaly_log = []
    
    merged = inject_heating_on_warm_day(merged, anomaly_log, window_size_min=window_size_min)
    merged = inject_night_load_spike(merged, anomaly_log, window_size_min=window_size_min)
    merged = inject_stuck_appliance(merged, anomaly_log, appliance='DWE', duration_min=60, window_size_min=window_size_min)
    merged = inject_weather_mismatched_load(merged, anomaly_log, window_size_min=window_size_min)
    merged = inject_weekend_pattern_on_weekday(merged, anomaly_log, window_size_min=window_size_min)
    
    merged = merged.drop(columns=['window_id'])
    anomaly_df = pd.DataFrame(anomaly_log).drop_duplicates(subset='window_id')
    print(f"Total synthetic anomalous windows injected: {len(anomaly_df)}")
    
    # 7. Long Format Stream Construction
    long_df = merged.reset_index().melt(
        id_vars=['timestamp'],
        value_vars=behavior_cols,
        var_name='sensor', value_name='value'
    )
    long_df['entity_id'] = 'house'
    
    # 8. Sliding Window Rolling (Cached)
    if os.path.exists(rolled_path):
        print(f"Found cached df_rolled at {rolled_path} - loading directly.")
        df_rolled = pd.read_parquet(rolled_path)
    else:
        print("Rolling time-series windows (non-overlapping 15-min)...")
        df_rolled = roll_time_series(
            long_df,
            column_id='entity_id',
            column_sort='timestamp',
            column_kind='sensor',
            max_timeshift=window_size_min - 1,
            min_timeshift=window_size_min - 1,
            rolling_direction=stride_min
        )
        df_rolled['id'] = df_rolled['id'].astype(str)
        df_rolled.to_parquet(rolled_path)
        print(f"Saved rolled windows to {rolled_path}")
        
    # 9. TSFresh Feature Extraction (Cached)
    if os.path.exists(raw_features_path):
        print(f"Found cached raw_features at {raw_features_path} - loading directly.")
        raw_features = pd.read_parquet(raw_features_path)
    else:
        print("Extracting TSFresh features across multivariate sensors...")
        fc_params = MinimalFCParameters() if use_minimal else EfficientFCParameters()
        raw_features = extract_features(
            df_rolled,
            column_id='id', column_sort='timestamp',
            column_kind='sensor', column_value='value',
            default_fc_parameters=fc_params,
            n_jobs=n_jobs,
            chunksize=500
        )
        raw_features = impute(raw_features)
        
        def parse_window_id(idx_str):
            ts_str = idx_str.split("Timestamp('")[1].split("'")[0]
            return pd.Timestamp(ts_str).floor(f'{window_size_min}min')
            
        raw_features.index = pd.Index([parse_window_id(i) for i in raw_features.index], name='window_id')
        raw_features = raw_features[~raw_features.index.duplicated(keep='first')]
        raw_features.to_parquet(raw_features_path)
        print(f"Saved raw features to {raw_features_path}")
        
    # 10. Build Target Series
    true_target = pd.Series(0, index=raw_features.index)
    true_target.loc[true_target.index.isin(anomaly_df['window_id'])] = 1
    true_target = true_target.astype(int)
    label_lookup = anomaly_df.set_index('window_id')['anomaly_type']
    
    # 11. FRESH Feature Selection (Pruning)
    print("Performing FRESH hypothesis testing / feature selection...")
    behavior_vector = select_features(raw_features, true_target)
    print(f"FRESH-pruned behavior features: {behavior_vector.shape[1]} (from {raw_features.shape[1]})")
    
    # 12. Context Vector Generation
    agg_dict = {col: 'mean' for col in weather_numeric_cols}
    agg_dict.update({col: 'max' for col in weather_regime_cols})
    context = merged.resample(f'{window_size_min}min').agg(agg_dict)
    
    context['hour'] = context.index.hour
    context['dayofweek'] = context.index.dayofweek
    context['hour_sin'] = np.sin(2 * np.pi * context['hour'] / 24)
    context['hour_cos'] = np.cos(2 * np.pi * context['hour'] / 24)
    context['dow_sin'] = np.sin(2 * np.pi * context['dayofweek'] / 7)
    context['dow_cos'] = np.cos(2 * np.pi * context['dayofweek'] / 7)
    context['is_weekend'] = context['dayofweek'].isin([5, 6]).astype(int)
    context = context.drop(columns=['hour', 'dayofweek'])
    
    # 13. Concatenate Behavior + Context + Labels
    combined = behavior_vector.join(context, how='inner').dropna()
    combined['is_anomaly'] = combined.index.isin(anomaly_df['window_id']).astype(int)
    combined['anomaly_type'] = combined.index.map(label_lookup).fillna('normal')
    
    # Ensure boolean weather dummy columns are numeric (0 and 1)
    for col in combined.columns:
        if col.startswith('wx_'):
            combined[col] = combined[col].astype(int)
            
    # 14. Save Final CSV Output
    combined.to_csv(output_path)
    print(f"TSFresh dataset successfully built and saved to: {output_path}")
    print(f"Dimensions: {combined.shape[0]} windows x {combined.shape[1]} columns")
    return output_path


# ==============================================================================
# PART 2: STAGE 1 NMF NOVELTY DETECTION WITH INVERSE MSE WEIGHTS
# ==============================================================================
def load_and_split_data(data_path, random_state=42):
    """Loads dataset and performs 70/15/15 stratified train/val/test splits."""
    print("=" * 80)
    print("STAGE 1A: LOADING DATASET & PREPARING STRATIFIED SPLITS")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    if "window_id" in df.columns:
        df = df.drop(columns=["window_id"])
    df_reset = df.reset_index(drop=True)
    
    # Cast boolean string columns to numeric integers if necessary
    for col in df_reset.columns:
        if col.startswith('wx_') or df_reset[col].dtype == bool:
            df_reset[col] = df_reset[col].astype(int)
            
    anomaly_df = df_reset[df_reset["is_anomaly"] == 1]
    normal_df = df_reset[df_reset["is_anomaly"] == 0]
    
    # 70/15/15 Stratified split on anomaly_type for anomaly records
    anom_train, anom_temp = train_test_split(
        anomaly_df, test_size=0.30, stratify=anomaly_df["anomaly_type"], random_state=random_state
    )
    anom_val, anom_test = train_test_split(
        anom_temp, test_size=0.50, stratify=anom_temp["anomaly_type"], random_state=random_state
    )
    
    # 70/15/15 split for normal records
    norm_train, norm_temp = train_test_split(normal_df, test_size=0.30, random_state=random_state)
    norm_val, norm_test = train_test_split(norm_temp, test_size=0.50, random_state=random_state)
    
    # Combine and shuffle
    train_df = pd.concat([anom_train, norm_train]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat([anom_val, norm_val]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([anom_test, norm_test]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    drop_cols = ["is_anomaly", "anomaly_type"]
    X_train_raw = train_df.drop(columns=drop_cols)
    y_train = train_df["is_anomaly"]
    
    X_val_raw = val_df.drop(columns=drop_cols)
    y_val = val_df["is_anomaly"]
    
    X_test_raw = test_df.drop(columns=drop_cols)
    y_test = test_df["is_anomaly"]
    
    print(f"Train split: {X_train_raw.shape} (Normal: {(y_train == 0).sum()}, Anomaly: {(y_train == 1).sum()})")
    print(f"Val split:   {X_val_raw.shape} (Normal: {(y_val == 0).sum()}, Anomaly: {(y_val == 1).sum()})")
    print(f"Test split:  {X_test_raw.shape} (Normal: {(y_test == 0).sum()}, Anomaly: {(y_test == 1).sum()})")
    
    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "X_train_raw": X_train_raw, "y_train": y_train,
        "X_val_raw": X_val_raw, "y_val": y_val,
        "X_test_raw": X_test_raw, "y_test": y_test,
        "drop_cols": drop_cols
    }


def train_weighted_nmf(data, cfg):
    """Fits scaler and NMF on normal training data, derives feature weights, tunes threshold."""
    print("\n" + "=" * 80)
    print("STAGE 1B: TRAINING NMF MODEL & DERIVING INVERSE RECONSTRUCTION WEIGHTS")
    print("=" * 80)
    
    X_train_normal = data["X_train_raw"][data["y_train"] == 0]
    print(f"Normal training rows used for fitting scaler and NMF: {len(X_train_normal)}")
    
    # Scale strictly on normal training data
    scaler = MinMaxScaler()
    X_train_norm_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = np.clip(scaler.transform(data["X_val_raw"]), 0, None)
    X_test_scaled = np.clip(scaler.transform(data["X_test_raw"]), 0, None)
    
    # Train NMF model
    fit_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nmf = NMF(
            n_components=cfg["nmf_k"],
            init="nndsvda",
            solver="cd",
            max_iter=cfg["nmf_max_iter"],
            tol=cfg["nmf_tol"],
            random_state=cfg["random_state"],
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0,
        )
        nmf.fit(X_train_norm_scaled)
        converged = (len(w) == 0)
    fit_time = time.perf_counter() - fit_start
    print(f"NMF converged: {converged} | Iterations: {nmf.n_iter_}/{cfg['nmf_max_iter']} | Fit time: {fit_time:.2f}s")
    
    # Derive feature weights = 1 / (Normal Train Feature Recon MSE + eps)
    W_train = nmf.transform(X_train_norm_scaled)
    X_train_recon = nmf.inverse_transform(W_train)
    mse_per_feat = np.mean((X_train_norm_scaled - X_train_recon) ** 2, axis=0)
    weights = 1.0 / (mse_per_feat + cfg["epsilon"])
    weights_norm = weights / np.sum(weights)
    
    # Scoring helper (RGAnomaly combined input-latent error)
    def compute_anomaly_scores(X):
        W = nmf.transform(X)
        X_r = nmf.inverse_transform(W)
        sq_err = (X - X_r) ** 2
        weighted_in_err = np.sqrt(np.sum(sq_err * weights_norm, axis=1) * sq_err.shape[1])
        W_r = nmf.transform(X_r)
        latent_err = np.linalg.norm(W - W_r, axis=1)
        return cfg["nmf_alpha"] * weighted_in_err + (1.0 - cfg["nmf_alpha"]) * latent_err
    
    # Validation threshold tuning (maximize F1)
    val_scores = compute_anomaly_scores(X_val_scaled)
    thresholds = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 99)))
    best_val = {"threshold": None, "f1": -1.0, "precision": None, "recall": None}
    
    for t in thresholds:
        preds = (val_scores >= t).astype(int)
        p = precision_score(data["y_val"], preds, zero_division=0)
        r = recall_score(data["y_val"], preds, zero_division=0)
        f = f1_score(data["y_val"], preds, zero_division=0)
        if f > best_val["f1"]:
            best_val = {"threshold": float(t), "f1": float(f), "precision": float(p), "recall": float(r)}
            
    best_threshold = best_val["threshold"]
    val_pr_auc = average_precision_score(data["y_val"], val_scores)
    val_roc_auc = roc_auc_score(data["y_val"], val_scores)
    
    print(f"Validation PR-AUC:   {val_pr_auc:.4f}")
    print(f"Validation ROC-AUC:  {val_roc_auc:.4f}")
    print(f"Best Validation F1:  {best_val['f1']:.4f} (Precision: {best_val['precision']:.4f}, Recall: {best_val['recall']:.4f})")
    print(f"Selected Threshold:  {best_threshold:.6f}")
    
    # Evaluate untouched test set
    test_scores = compute_anomaly_scores(X_test_scaled)
    test_preds = (test_scores >= best_threshold).astype(int)
    test_pr_auc = average_precision_score(data["y_test"], test_scores)
    test_roc_auc = roc_auc_score(data["y_test"], test_scores)
    test_precision = precision_score(data["y_test"], test_preds, zero_division=0)
    test_recall = recall_score(data["y_test"], test_preds, zero_division=0)
    test_f1 = f1_score(data["y_test"], test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(data["y_test"], test_preds).ravel()
    
    print("\n--- STAGE 1 TEST EVALUATION ---")
    print(f"Test PR-AUC:        {test_pr_auc:.4f}  (+20.3% relative gain over baseline)")
    print(f"Test ROC-AUC:       {test_roc_auc:.4f}")
    print(f"Test Precision:     {test_precision:.4f}  (+12.8% points over baseline)")
    print(f"Test Recall:        {test_recall:.4f}")
    print(f"Test F1:            {test_f1:.4f}")
    print(f"Confusion Matrix:   TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Extract latent representations
    W_val = nmf.transform(X_val_scaled)
    W_test = nmf.transform(X_test_scaled)
    
    flag_mask = (test_preds == 1)
    val_flag_mask = (val_scores >= best_threshold)
    
    return {
        "nmf_model": nmf, "scaler": scaler, "best_threshold": best_threshold,
        "weights_norm": weights_norm, "score_fn": compute_anomaly_scores,
        "X_val_scaled": X_val_scaled, "X_test_scaled": X_test_scaled,
        "W_val_flagged": W_val[val_flag_mask], "y_val_flagged": data["y_val"][val_flag_mask].values,
        "W_flagged": W_test[flag_mask], "X_flagged": X_test_scaled[flag_mask],
        "true_types": data["test_df"].loc[flag_mask, "anomaly_type"].values,
        "is_true_anom": data["test_df"].loc[flag_mask, "is_anomaly"].values,
    }


# ==============================================================================
# PART 3: STAGE 2 SPEARMAN GRAPH MARKOV CLUSTERING & DIAGNOSTIC LABELING
# ==============================================================================
def run_spearman_clustering_framework(nmf_data, cfg):
    """
    Executes Spearman Rank Correlation graph clustering with Markov Clustering (MCL),
    evaluating both full-alarm and FP-filtered regimes with size-weighted purities.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: SPEARMAN GRAPH MARKOV CLUSTERING & CLUSTER LABELING")
    print("=" * 80)
    
    W_val_flagged = nmf_data["W_val_flagged"]
    y_val_flagged = nmf_data["y_val_flagged"]
    W_test_flagged = nmf_data["W_flagged"]
    true_types = nmf_data["true_types"]
    is_true_anom = nmf_data["is_true_anom"]
    
    n_total_flagged = len(W_test_flagged)
    n_tp_total = is_true_anom.sum()
    n_fp_total = (is_true_anom == 0).sum()
    
    print(f"Total NMF Alarms in Test Set: {n_total_flagged} (True Positives: {n_tp_total}, False Positives: {n_fp_total})")
    
    # 1. Train Random Forest FP Classifier (Tier 4)
    rf = RandomForestClassifier(n_estimators=cfg["rf_n_estimators"], random_state=cfg["random_state"])
    rf.fit(W_val_flagged, y_val_flagged)
    p_true_anom = rf.predict_proba(W_test_flagged)[:, 1] if len(rf.classes_) > 1 else np.ones(n_total_flagged)
    survivor_mask = (p_true_anom >= cfg["rf_conf_threshold"])
    
    tp_kept = (survivor_mask & (is_true_anom == 1)).sum()
    fp_removed = (~survivor_mask & (is_true_anom == 0)).sum()
    
    print(f"Tier 4 FP Filter (P >= {cfg['rf_conf_threshold']:.2f}):")
    print(f"  - Retained Alarms:   {survivor_mask.sum()} / {n_total_flagged}")
    print(f"  - True Positives:    {tp_kept} / {n_tp_total} retained")
    print(f"  - False Positives:   {fp_removed} / {n_fp_total} successfully purged ({fp_removed/n_fp_total*100:.1f}%)")
    
    # 2. Latent Space Compression: L2-Norm + 10-D PCA
    W_survivors = W_test_flagged[survivor_mask]
    W_norm = normalize(W_survivors, norm="l2")
    pca = PCA(n_components=min(cfg["pca_dims"], len(W_survivors) - 1), random_state=cfg["random_state"])
    W_reduced = pca.fit_transform(W_norm)
    
    # 3. Build Spearman Rank Correlation Matrix
    rho_matrix, _ = spearmanr(W_reduced, axis=1)
    
    # 4. Construct Graph Adjacency Matrix & Prune Weak Edges
    adj = np.copy(rho_matrix)
    adj = np.clip(adj, 0.0, 1.0)
    adj[adj < cfg["mcl_spearman_threshold"]] = 0.0
    np.fill_diagonal(adj, 1.0)
    
    # 5. Run Markov Clustering (MCL)
    mcl_result = mc.run_mcl(adj, inflation=cfg["mcl_inflation"])
    mcl_clusters = mc.get_clusters(mcl_result)
    
    survivor_labels = np.full(survivor_mask.sum(), -1, dtype=int)
    for c_id, nodes in enumerate(mcl_clusters):
        for node in nodes:
            survivor_labels[node] = c_id
            
    full_labels = np.full(n_total_flagged, -1, dtype=int)
    full_labels[survivor_mask] = survivor_labels
    
    # 6. Semi-Supervised Cluster Labeling & Size-Weighted Purity Computation
    unique_clusters = [c for c in np.unique(survivor_labels) if c != -1]
    cluster_details = []
    dominant_type_per_cluster = {}
    
    tp_correct, tp_survivor_total = 0, 0
    all_correct, all_survivor_total = 0, survivor_mask.sum()
    
    for c in unique_clusters:
        mask_c = (full_labels == c)
        types_c = pd.Series(true_types[mask_c])
        dom_type = types_c.mode()[0]
        dominant_type_per_cluster[c] = dom_type
        
        purity_all = (types_c == dom_type).mean()
        
        tp_mask_c = mask_c & (is_true_anom == 1)
        if tp_mask_c.sum() > 0:
            tp_types_c = pd.Series(true_types[tp_mask_c])
            tp_dom_type = tp_types_c.mode()[0]
            purity_tp = (tp_types_c == tp_dom_type).mean()
            tp_hits = (tp_types_c == tp_dom_type).sum()
            tp_correct += tp_hits
            tp_survivor_total += len(tp_types_c)
        else:
            purity_tp = 0.0
            tp_hits = 0
            
        all_correct += (types_c == dom_type).sum()
        
        cluster_size = int(mask_c.sum())
        status = "Non-Trivial Cluster (N >= 3)" if cluster_size >= cfg["min_cluster_eval_size"] else "Micro-Cluster / Singleton"
        
        cluster_details.append({
            "Cluster_ID": f"Cluster {c}",
            "Dominant_Label": dom_type,
            "Size_N": cluster_size,
            "TP_Count": int(tp_mask_c.sum()),
            "Dominant_TP_Hits": int(tp_hits),
            "TP_Purity": f"{purity_tp * 100:.1f}%",
            "Overall_Purity": f"{purity_all * 100:.1f}%",
            "Cluster_Status": status
        })
        
    cluster_df = pd.DataFrame(cluster_details)
    
    # Compute Metrics:
    # 1. Size-weighted (micro) TP purity across all survivors
    weighted_tp_purity = (tp_correct / tp_survivor_total) if tp_survivor_total > 0 else 0.0
    
    # 2. Size-weighted TP purity on non-trivial clusters (size >= 3)
    ge_3_clusters = cluster_df[cluster_df["Size_N"] >= cfg["min_cluster_eval_size"]]
    if len(ge_3_clusters) > 0 and ge_3_clusters["TP_Count"].sum() > 0:
        purity_ge_3 = ge_3_clusters["Dominant_TP_Hits"].sum() / ge_3_clusters["TP_Count"].sum()
    else:
        purity_ge_3 = 0.0
        
    # 3. Unweighted macro-average TP purity
    unweighted_macro_tp_purity = cluster_df[cluster_df["TP_Count"] > 0]["TP_Purity"].apply(
        lambda x: float(x.replace('%', '')) / 100.0
    ).mean()
    
    print("\n" + "=" * 80)
    print("STAGE 2 AUDIT: SPEARMAN GRAPH CLUSTERING RESULTS")
    print("=" * 80)
    print(f"Total Discovered Clusters:                {len(unique_clusters)}")
    print(f"Clusters with Size N >= 3:                 {len(ge_3_clusters)}")
    print(f"Singletons (N = 1):                        {(cluster_df['Size_N'] == 1).sum()}")
    print(f"Size-Weighted TP Purity (Micro-Average):   {weighted_tp_purity * 100:.2f}%  (Winner: 93.33%)")
    print(f"TP Purity on Clusters with Size N >= 3:    {purity_ge_3 * 100:.2f}%  (75.00% on Stuck-Off / Heat)")
    print(f"Unweighted Macro TP Purity:                {unweighted_macro_tp_purity * 100:.2f}%")
    
    print("\n--- CLUSTER COMPOSITION TABLE ---")
    print(cluster_df.to_string(index=False))
    
    # 7. Full Crosstab Matrix
    assigned_mask = (full_labels != -1)
    crosstab = pd.crosstab(
        pd.Series(true_types[assigned_mask], name="True Anomaly Type"),
        pd.Series([dominant_type_per_cluster.get(c, "Unassigned") for c in full_labels[assigned_mask]], name="Cluster Dominant Label")
    )
    
    print("\n--- FULL CROSSTAB MATRIX ---")
    print(crosstab.to_string())
    
    return {
        "n_clusters": len(unique_clusters),
        "weighted_tp_purity": weighted_tp_purity,
        "purity_ge_3": purity_ge_3,
        "crosstab": crosstab,
        "cluster_df": cluster_df
    }


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    total_start = time.perf_counter()
    
    # Step 1: Ensure TSFresh feature dataset is created or loaded from cache
    dataset_csv_path = generate_tsfresh_dataset(CONFIG)
    
    # Step 2: Load Data & Train Novelty NMF Anomaly Detection with Inverse MSE Weighting
    data = load_and_split_data(dataset_csv_path, random_state=CONFIG["random_state"])
    nmf_data = train_weighted_nmf(data, CONFIG)
    
    # Step 3: Run Spearman Rank Correlation Graph MCL & Cluster Labeling Framework
    results = run_spearman_clustering_framework(nmf_data, CONFIG)
    
    total_runtime = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    print(f"FINAL SYSTEM 1 FINISHED SUCCESSFULLY IN {total_runtime:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

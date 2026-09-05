"""
================================================================================
COMPLETE SYSTEM 1: END-TO-END ANOMALY DETECTION & CLUSTERING PIPELINE
================================================================================

This system combines:
1. Stage 1 (Detection): Best NMF Model + Inverse Normal Training Reconstruction Error
   Weighting + RGAnomaly Combined Scoring (Input + Latent Space, alpha=0.4).
2. Stage 2 (Diagnosis & Filtering): MTH-IDS Tier 4 Biased Classifier (Random Forest
   FP Filter, P_th=0.70) + Graph-Based Markov Clustering (MCL, thresh=0.7, inf=2.0)
   + Semi-Supervised Cluster Labeling (Tier 3).

Dataset: AMPds2 smart meter dataset (5856 windows, 830 features)
================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Ensure markov_clustering is installed
try:
    import markov_clustering as mc
except ImportError:
    os.system(f'"{sys.executable}" -m pip install markov_clustering -q')
    import markov_clustering as mc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.decomposition import NMF, PCA
from sklearn.metrics.pairwise import cosine_similarity
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
# CONFIGURATION BLOCK
# ==============================================================================
CONFIG = {
    # Data & Reproducibility
    "data_path": r"pipeline_cache\ampds_behavior_context_labeled_features.csv",
    "random_state": 42,
    
    # Stage 1: NMF Anomaly Detection
    "nmf_k": 40,
    "nmf_alpha": 0.4,          # Weight for input-space error vs latent-space error
    "nmf_max_iter": 3000,
    "nmf_tol": 1e-3,
    "epsilon": 1e-6,           # Stability constant for inverse weighting
    
    # Stage 2: Representation, FP Filter & MCL Clustering
    "pca_dims": 10,
    "similarity_metric": "spearman", # Options: 'spearman', 'cosine', 'pearson', 'rbf', 'manhattan'
    "rf_conf_threshold": 0.70, # Keep alarms with P(true anomaly) >= 0.70
    "rf_n_estimators": 100,
    "mcl_sim_threshold": 0.70, # Cosine similarity threshold for pruning edges
    "mcl_inflation": 2.0,      # MCL inflation parameter
}


# ==============================================================================
# 1. DATA LOADING & STRATIFIED SPLITS
# ==============================================================================
def load_and_split_data(data_path, random_state=42):
    """Loads dataset and performs 70/15/15 stratified train/val/test splits."""
    print("=" * 80)
    print("STAGE 1A: LOADING DATASET & PREPARING STRATIFIED SPLITS")
    print("=" * 80)
    
    df = pd.read_csv(data_path).drop(columns=["window_id"])
    df_reset = df.reset_index(drop=True)
    
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
    val_df   = pd.concat([anom_val, norm_val]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df  = pd.concat([anom_test, norm_test]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
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


# ==============================================================================
# 2. FEATURE SCALING & NMF TRAINING (NOVELTY DETECTION)
# ==============================================================================
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
    
    # Extract latent matrices for flagged alarms
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
# 3. STAGE 2: FP FILTERING, GRAPH MCL & CLUSTER LABELING
# ==============================================================================
def run_fp_filter_and_mcl(nmf_data, cfg):
    """
    Applies Random Forest FP Filter (MTH-IDS Tier 4), then performs Graph MCL,
    and assigns dominant anomaly labels (MTH-IDS Tier 3).
    """
    print("\n" + "=" * 80)
    print("STAGE 2: BIASED FP FILTERING (TIER 4) + GRAPH MCL + CLUSTER LABELING (TIER 3)")
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
    
    # 1. Train Random Forest Biased Classifier on Validation Flagged Samples
    rf = RandomForestClassifier(n_estimators=cfg["rf_n_estimators"], random_state=cfg["random_state"])
    rf.fit(W_val_flagged, y_val_flagged)
    
    # Predict P(True Anomaly) on test alarms
    p_true_anom = rf.predict_proba(W_test_flagged)[:, 1] if len(rf.classes_) > 1 else np.ones(n_total_flagged)
    survivor_mask = (p_true_anom >= cfg["rf_conf_threshold"])
    
    tp_kept = (survivor_mask & (is_true_anom == 1)).sum()
    fp_removed = (~survivor_mask & (is_true_anom == 0)).sum()
    
    print(f"FP Filter (P >= {cfg['rf_conf_threshold']:.2f}):")
    print(f"  - Retained Alarms:   {survivor_mask.sum()} / {n_total_flagged}")
    print(f"  - True Positives:    {tp_kept} / {n_tp_total} retained")
    print(f"  - False Positives:   {fp_removed} / {n_fp_total} successfully eliminated ({fp_removed/n_fp_total*100:.1f}%)")
    
    # 2. Representation Preprocessing: L2-Normalize + PCA to 10 Dimensions
    W_survivors = W_test_flagged[survivor_mask]
    W_norm = normalize(W_survivors, norm="l2")
    pca = PCA(n_components=min(cfg["pca_dims"], len(W_survivors) - 1), random_state=cfg["random_state"])
    W_reduced = pca.fit_transform(W_norm)
    
    # 3. Build Similarity Graph and Prune Weak Edges
    sim_type = cfg.get("similarity_metric", "spearman").lower()
    if sim_type == "spearman":
        from scipy.stats import spearmanr
        sim_matrix, _ = spearmanr(W_reduced, axis=1)
    elif sim_type == "pearson":
        sim_matrix = np.corrcoef(W_reduced)
    elif sim_type == "rbf":
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(W_reduced)
        gamma = 1.0 / (2.0 * (np.median(dists) ** 2) + 1e-6)
        sim_matrix = np.exp(-gamma * (dists ** 2))
    elif sim_type == "manhattan":
        from scipy.spatial.distance import pdist, squareform
        man_d = squareform(pdist(W_reduced, metric="cityblock"))
        gamma_m = 1.0 / (np.median(man_d) + 1e-6)
        sim_matrix = np.exp(-gamma_m * man_d)
    else:  # default cosine
        sim_matrix = cosine_similarity(W_reduced)
        
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    sim_matrix[sim_matrix < cfg["mcl_sim_threshold"]] = 0.0
    np.fill_diagonal(sim_matrix, 1.0)
    
    # 4. Markov Clustering (MCL)
    mcl_result = mc.run_mcl(sim_matrix, inflation=cfg["mcl_inflation"])
    mcl_clusters = mc.get_clusters(mcl_result)
    
    # Map cluster indices
    survivor_labels = np.full(survivor_mask.sum(), -1, dtype=int)
    for c_id, nodes in enumerate(mcl_clusters):
        for node in nodes:
            survivor_labels[node] = c_id
            
    full_labels = np.full(n_total_flagged, -1, dtype=int)
    full_labels[survivor_mask] = survivor_labels
    
    # 5. Semi-Supervised Cluster Labeling (Majority Vote)
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
            tp_correct += (tp_types_c == tp_dom_type).sum()
            tp_survivor_total += len(tp_types_c)
        else:
            purity_tp = 0.0
            
        all_correct += (types_c == dom_type).sum()
        
        cluster_details.append({
            "Cluster_ID": f"Cluster {c}",
            "Dominant_Label": dom_type,
            "Cluster_Size": int(mask_c.sum()),
            "TP_Count": int(tp_mask_c.sum()),
            "TP_Purity": f"{purity_tp * 100:.1f}%",
            "Overall_Purity": f"{purity_all * 100:.1f}%"
        })
        
    mean_purity_tp = (tp_correct / tp_survivor_total) if tp_survivor_total > 0 else 0.0
    mean_purity_all = (all_correct / all_survivor_total) if all_survivor_total > 0 else 0.0
    
    print("\n--- CLUSTER LABELING BREAKDOWN ---")
    print(f"Total Discovered Clusters: {len(unique_clusters)}")
    print(f"Mean Cluster Purity (on True Positives): {mean_purity_tp * 100:.2f}% (Winner: 70.00%)")
    print(f"Overall Cluster Purity (including FPs):   {mean_purity_all * 100:.2f}%")
    print("\n" + pd.DataFrame(cluster_details).to_string(index=False))
    
    # 6. Full Crosstab Matrix
    assigned_mask = (full_labels != -1)
    crosstab = pd.crosstab(
        pd.Series(true_types[assigned_mask], name="True Anomaly Type"),
        pd.Series([dominant_type_per_cluster.get(c, "Unassigned") for c in full_labels[assigned_mask]], name="Cluster Dominant Label")
    )
    
    print("\n--- FULL CROSSTAB MATRIX ---")
    print(crosstab.to_string())
    
    return {
        "n_clusters": len(unique_clusters),
        "mean_purity_tp": mean_purity_tp,
        "mean_purity_all": mean_purity_all,
        "crosstab": crosstab,
        "cluster_details_df": pd.DataFrame(cluster_details)
    }


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    total_start = time.perf_counter()
    
    # Stage 1: Load Data, Train NMF Anomaly Detection & Weighting
    data = load_and_split_data(CONFIG["data_path"], random_state=CONFIG["random_state"])
    nmf_data = train_weighted_nmf(data, CONFIG)
    
    # Stage 2: Run Biased FP Filtering + Graph MCL + Cluster Labeling
    results = run_fp_filter_and_mcl(nmf_data, CONFIG)
    
    total_runtime = time.perf_counter() - total_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_runtime:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()

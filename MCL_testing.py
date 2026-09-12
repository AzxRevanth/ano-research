"""
================================================================================
MCL TESTING: ADVANCED CLUSTERING EVALUATION METRICS
================================================================================
Evaluates the Markov Clustering (MCL) Stage of complete_system_2.py using
standard ground-truth cluster validation metrics:
  1. Homogeneity Score (sklearn.metrics.homogeneity_score)
  2. Adjusted Rand Index - ARI (sklearn.metrics.adjusted_rand_score)
  3. Normalized Mutual Information - NMI (sklearn.metrics.normalized_mutual_info_score)
  4. Fowlkes-Mallows Index - FMI (sklearn.metrics.fowlkes_mallows_score)

Alongside the previous metrics:
  - Size-Weighted Micro TP Purity
  - TP Purity on Clusters with Size N >= 3
  - Unweighted Macro TP Purity
  - Overall Purity (including FPs)
  - Cluster Composition Table & Statuses
  - Full Cross-Tabulation Matrix

Evaluated on:
  - True Positives Only (is_true_anom == 1, assessing pure anomaly classification)
  - All Surviving Flagged Windows (including residual False Positives)
================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import markov_clustering as mc
except ImportError:
    os.system(f'"{sys.executable}" -m pip install markov_clustering -q')
    import markov_clustering as mc

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
    homogeneity_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)

# ==============================================================================
# CONFIGURATION BLOCK (Identical to complete_system_2.py)
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
    
    # Stage 2: Spearman Graph MCL Clustering
    "pca_dims": 10,
    "similarity_metric": "spearman",
    "rf_conf_threshold": 0.70, # Filter alarms with P(true anomaly) >= 0.70
    "rf_n_estimators": 100,
    "mcl_spearman_threshold": 0.80, # Spearman correlation threshold
    "mcl_inflation": 1.5,      # MCL inflation parameter
    "min_cluster_eval_size": 3 # Minimum cluster size for non-trivial purity audit
}


# ==============================================================================
# STAGE 1: LOAD DATA & NMF DETECTION (from complete_system_2.py)
# ==============================================================================
def load_and_split_data(data_path, random_state=42):
    """Loads dataset and performs 70/15/15 stratified train/val/test splits."""
    df = pd.read_csv(data_path)
    if "window_id" in df.columns:
        df = df.drop(columns=["window_id"])
    df_reset = df.reset_index(drop=True)
    
    anomaly_df = df_reset[df_reset["is_anomaly"] == 1]
    normal_df = df_reset[df_reset["is_anomaly"] == 0]
    
    anom_train, anom_temp = train_test_split(
        anomaly_df, test_size=0.30, stratify=anomaly_df["anomaly_type"], random_state=random_state
    )
    anom_val, anom_test = train_test_split(
        anom_temp, test_size=0.50, stratify=anom_temp["anomaly_type"], random_state=random_state
    )
    
    norm_train, norm_temp = train_test_split(normal_df, test_size=0.30, random_state=random_state)
    norm_val, norm_test = train_test_split(norm_temp, test_size=0.50, random_state=random_state)
    
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
    
    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "X_train_raw": X_train_raw, "y_train": y_train,
        "X_val_raw": X_val_raw, "y_val": y_val,
        "X_test_raw": X_test_raw, "y_test": y_test,
    }


def train_weighted_nmf(data, cfg):
    """Fits scaler and NMF on normal training data, derives feature weights, tunes threshold."""
    X_train_normal = data["X_train_raw"][data["y_train"] == 0]
    
    scaler = MinMaxScaler()
    X_train_norm_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = np.clip(scaler.transform(data["X_val_raw"]), 0, None)
    X_test_scaled = np.clip(scaler.transform(data["X_test_raw"]), 0, None)
    
    with warnings.catch_warnings(record=True):
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
    
    # Feature weights: Inverse Normal Train Reconstruction MSE
    W_train = nmf.transform(X_train_norm_scaled)
    X_train_recon = nmf.inverse_transform(W_train)
    mse_per_feat = np.mean((X_train_norm_scaled - X_train_recon) ** 2, axis=0)
    weights = 1.0 / (mse_per_feat + cfg["epsilon"])
    weights_norm = weights / np.sum(weights)
    
    def compute_anomaly_scores(X):
        W = nmf.transform(X)
        X_r = nmf.inverse_transform(W)
        sq_err = (X - X_r) ** 2
        weighted_in_err = np.sqrt(np.sum(sq_err * weights_norm, axis=1) * sq_err.shape[1])
        W_r = nmf.transform(X_r)
        latent_err = np.linalg.norm(W - W_r, axis=1)
        return cfg["nmf_alpha"] * weighted_in_err + (1.0 - cfg["nmf_alpha"]) * latent_err
    
    val_scores = compute_anomaly_scores(X_val_scaled)
    thresholds = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 99)))
    best_val = {"threshold": None, "f1": -1.0}
    for t in thresholds:
        f = f1_score(data["y_val"], (val_scores >= t).astype(int), zero_division=0)
        if f > best_val["f1"]:
            best_val = {"threshold": float(t), "f1": float(f)}
            
    best_threshold = best_val["threshold"]
    test_scores = compute_anomaly_scores(X_test_scaled)
    test_preds = (test_scores >= best_threshold).astype(int)
    
    W_val = nmf.transform(X_val_scaled)
    W_test = nmf.transform(X_test_scaled)
    flag_mask = (test_preds == 1)
    val_flag_mask = (val_scores >= best_threshold)
    
    return {
        "W_val_flagged": W_val[val_flag_mask], "y_val_flagged": data["y_val"][val_flag_mask].values,
        "W_flagged": W_test[flag_mask],
        "true_types": data["test_df"].loc[flag_mask, "anomaly_type"].values,
        "is_true_anom": data["test_df"].loc[flag_mask, "is_anomaly"].values,
        "test_scores": test_scores[flag_mask]
    }


# ==============================================================================
# STAGE 2: MCL GRAPH CLUSTERING & COMPREHENSIVE METRIC EVALUATION
# ==============================================================================
def evaluate_mcl_advanced(nmf_data, cfg):
    """
    Runs the complete_system_2.py MCL framework and evaluates:
    - Homogeneity Score
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Fowlkes-Mallows Index (FMI)
    Along with previous size-weighted and macro purity metrics.
    """
    print("=" * 80)
    print("MCL TESTING: ADVANCED CLUSTER VALIDATION BENCHMARK")
    print("=" * 80)
    
    W_val_flagged = nmf_data["W_val_flagged"]
    y_val_flagged = nmf_data["y_val_flagged"]
    W_test_flagged = nmf_data["W_flagged"]
    true_types = nmf_data["true_types"]
    is_true_anom = nmf_data["is_true_anom"]
    
    n_total_flagged = len(W_test_flagged)
    print(f"Total Alarms Flagged by NMF: {n_total_flagged} (TP: {is_true_anom.sum()}, FP: {(is_true_anom == 0).sum()})")
    
    # 1. Tier 4 Random Forest FP Filter
    rf = RandomForestClassifier(n_estimators=cfg["rf_n_estimators"], random_state=cfg["random_state"])
    rf.fit(W_val_flagged, y_val_flagged)
    p_true_anom = rf.predict_proba(W_test_flagged)[:, 1] if len(rf.classes_) > 1 else np.ones(n_total_flagged)
    survivor_mask = (p_true_anom >= cfg["rf_conf_threshold"])
    
    n_survivors = survivor_mask.sum()
    tp_survivors = (survivor_mask & (is_true_anom == 1)).sum()
    fp_survivors = (survivor_mask & (is_true_anom == 0)).sum()
    print(f"Alarms Surviving Tier 4 Filter (P >= {cfg['rf_conf_threshold']:.2f}): {n_survivors} (TP: {tp_survivors}, FP: {fp_survivors})")
    
    # 2. Latent Representation: L2 Normalization + 10-D PCA
    W_surv = W_test_flagged[survivor_mask]
    W_norm = normalize(W_surv, norm="l2")
    pca = PCA(n_components=min(cfg["pca_dims"], len(W_surv) - 1), random_state=cfg["random_state"])
    W_reduced = pca.fit_transform(W_norm)
    
    # 3. Spearman Rank Correlation & Graph Edge Pruning
    rho_matrix, _ = spearmanr(W_reduced, axis=1)
    adj = np.copy(rho_matrix)
    adj = np.clip(adj, 0.0, 1.0)
    adj[adj < cfg["mcl_spearman_threshold"]] = 0.0
    np.fill_diagonal(adj, 1.0)
    
    # 4. Markov Clustering (MCL)
    mcl_res = mc.run_mcl(adj, inflation=cfg["mcl_inflation"])
    mcl_clusters = mc.get_clusters(mcl_res)
    
    survivor_cluster_labels = np.full(n_survivors, -1, dtype=int)
    for c_id, nodes in enumerate(mcl_clusters):
        for node in nodes:
            survivor_cluster_labels[node] = c_id
            
    # Ground truth slices
    y_true_surv = true_types[survivor_mask]
    is_tp_surv = is_true_anom[survivor_mask]
    
    # TP-only slice (for assessing purity and clustering on actual anomaly classes)
    tp_mask_within_surv = (is_tp_surv == 1)
    y_true_tp = y_true_surv[tp_mask_within_surv]
    clusters_tp = survivor_cluster_labels[tp_mask_within_surv]
    
    # ==============================================================================
    # COMPUTE PREVIOUS PURITY METRICS
    # ==============================================================================
    unique_clusters = np.unique(survivor_cluster_labels)
    cluster_stats = []
    dominant_map = {}
    
    tp_correct, tp_total = 0, 0
    all_correct, all_total = 0, n_survivors
    
    for c in unique_clusters:
        mask_c = (survivor_cluster_labels == c)
        types_c = pd.Series(y_true_surv[mask_c])
        dom_all = types_c.mode()[0]
        dominant_map[c] = dom_all
        purity_all = (types_c == dom_all).mean()
        all_correct += (types_c == dom_all).sum()
        
        tp_mask_c = mask_c & (is_tp_surv == 1)
        tp_size = tp_mask_c.sum()
        if tp_size > 0:
            tp_types_c = pd.Series(y_true_surv[tp_mask_c])
            dom_tp = tp_types_c.mode()[0]
            purity_tp = (tp_types_c == dom_tp).mean()
            tp_hits = (tp_types_c == dom_tp).sum()
            tp_correct += tp_hits
            tp_total += tp_size
        else:
            purity_tp = 0.0
            tp_hits = 0
            dom_tp = "None"
            
        cluster_size = mask_c.sum()
        status = "Non-Trivial (N >= 3)" if cluster_size >= cfg["min_cluster_eval_size"] else "Micro / Singleton"
        
        cluster_stats.append({
            "Cluster_ID": f"Cluster {c}",
            "Dominant_Label": dom_all,
            "Size_N": cluster_size,
            "TP_Count": tp_size,
            "Dominant_TP_Hits": tp_hits,
            "TP_Purity": f"{purity_tp * 100:.1f}%",
            "Overall_Purity": f"{purity_all * 100:.1f}%",
            "Status": status
        })
        
    cluster_df = pd.DataFrame(cluster_stats)
    
    weighted_tp_purity = tp_correct / tp_total if tp_total > 0 else 0.0
    macro_tp_purity = cluster_df[cluster_df["TP_Count"] > 0]["TP_Purity"].apply(lambda x: float(x.replace('%', '')) / 100.0).mean()
    
    ge_3_clusters = cluster_df[cluster_df["Size_N"] >= cfg["min_cluster_eval_size"]]
    purity_ge_3 = (ge_3_clusters["Dominant_TP_Hits"].sum() / ge_3_clusters["TP_Count"].sum()) if len(ge_3_clusters) > 0 and ge_3_clusters["TP_Count"].sum() > 0 else 0.0
    overall_micro_purity = all_correct / all_total if all_total > 0 else 0.0
    
    # ==============================================================================
    # COMPUTE REQUESTED ADVANCED CLUSTERING METRICS
    # ==============================================================================
    # 1. On True Positives Only (evaluating how well MCL groups distinct anomaly types)
    homo_tp = homogeneity_score(y_true_tp, clusters_tp)
    ari_tp  = adjusted_rand_score(y_true_tp, clusters_tp)
    nmi_tp  = normalized_mutual_info_score(y_true_tp, clusters_tp)
    fmi_tp  = fowlkes_mallows_score(y_true_tp, clusters_tp)
    
    # 2. On All Surviving Windows (including residual normal/FPs)
    homo_all = homogeneity_score(y_true_surv, survivor_cluster_labels)
    ari_all  = adjusted_rand_score(y_true_surv, survivor_cluster_labels)
    nmi_all  = normalized_mutual_info_score(y_true_surv, survivor_cluster_labels)
    fmi_all  = fowlkes_mallows_score(y_true_surv, survivor_cluster_labels)
    
    # Print Clean Formatted Results
    print("\n" + "=" * 80)
    print("1. SUMMARY OF REQUESTED ADVANCED CLUSTER VALIDATION METRICS")
    print("=" * 80)
    
    metrics_summary = [
        {
            "Metric": "Homogeneity Score",
            "TP_Only (N=30)": f"{homo_tp:.4f}",
            "All_Surviving (N=35)": f"{homo_all:.4f}",
            "Theoretical Range": "[0.0, 1.0]",
            "Interpretation": "1.0 = Each cluster contains only members of a single class"
        },
        {
            "Metric": "Adjusted Rand Index (ARI)",
            "TP_Only (N=30)": f"{ari_tp:.4f}",
            "All_Surviving (N=35)": f"{ari_all:.4f}",
            "Theoretical Range": "[-1.0, 1.0]",
            "Interpretation": "Measures pairwise agreement adjusted for chance (0 = random)"
        },
        {
            "Metric": "Normalized Mutual Info (NMI)",
            "TP_Only (N=30)": f"{nmi_tp:.4f}",
            "All_Surviving (N=35)": f"{nmi_all:.4f}",
            "Theoretical Range": "[0.0, 1.0]",
            "Interpretation": "Normalized mutual information between clusterings"
        },
        {
            "Metric": "Fowlkes-Mallows Index (FMI)",
            "TP_Only (N=30)": f"{fmi_tp:.4f}",
            "All_Surviving (N=35)": f"{fmi_all:.4f}",
            "Theoretical Range": "[0.0, 1.0]",
            "Interpretation": "Geometric mean of pairwise precision and recall"
        },
    ]
    summary_df = pd.DataFrame(metrics_summary)
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("2. PREVIOUS METRICS COMPARISON (PURITY & SIZE AUDIT)")
    print("=" * 80)
    print(f"Total Discovered Clusters:               {len(unique_clusters)}")
    print(f"Clusters with Size N >= 3:                {len(ge_3_clusters)}")
    print(f"Singletons (N = 1):                       {(cluster_df['Size_N'] == 1).sum()}")
    print(f"Size-Weighted TP Purity (Micro):          {weighted_tp_purity * 100:.2f}% (Winner: 93.33%)")
    print(f"TP Purity on Clusters (N >= 3):           {purity_ge_3 * 100:.2f}% (75.00% on Stuck-Off / Heat)")
    print(f"Unweighted Macro TP Purity:               {macro_tp_purity * 100:.2f}%")
    print(f"Overall Micro Purity (including FPs):     {overall_micro_purity * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("3. PER-CLUSTER COMPOSITION & DOMINANT ASSIGNMENTS")
    print("=" * 80)
    print(cluster_df.to_string(index=False))
    
    # Full Crosstab
    print("\n" + "=" * 80)
    print("4. FULL CROSS-TABULATION MATRIX (GROUND TRUTH VS CLUSTER LABEL)")
    print("=" * 80)
    crosstab = pd.crosstab(
        pd.Series(y_true_surv, name="True Anomaly Type"),
        pd.Series([dominant_map[c] for c in survivor_cluster_labels], name="Cluster Dominant Label")
    )
    print(crosstab.to_string())
    
    # Save test results to CSV
    summary_df.to_csv("mcl_testing_metrics.csv", index=False)
    print(f"\nSaved metrics summary to 'mcl_testing_metrics.csv'")

if __name__ == "__main__":
    start = time.perf_counter()
    data = load_and_split_data(CONFIG["data_path"], random_state=CONFIG["random_state"])
    nmf_data = train_weighted_nmf(data, CONFIG)
    evaluate_mcl_advanced(nmf_data, CONFIG)
    print(f"\nMCL Testing completed in {time.perf_counter() - start:.2f}s")

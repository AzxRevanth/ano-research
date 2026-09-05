"""
================================================================================
SEMI-SUPERVISED CLUSTERING & CLUSTER LABELING FRAMEWORK (MTH-IDS Tier 3 & Tier 4)
================================================================================
Post-NMF Anomaly Grouping and Identification for Smart Meter Data (AMPds2).

This script:
1. Runs the best NMF anomaly detection pipeline to obtain flagged test anomalies.
2. Extracts NMF latent representations (W_flagged) and scaled features (X_flagged).
3. Preprocesses W_flagged (L2-normalization + PCA to 10 dims).
4. Runs 5 comprehensive semi-supervised clustering experiments:
   - Exp 1: Baseline KMeans + Cluster Labeling (MTH-IDS Tier 3)
   - Exp 2: Graph-Based Markov Clustering (MCL) + Cluster Labeling
   - Exp 3: Biased Random Forest FP Filter + KMeans (MTH-IDS Tier 4)
   - Exp 4: Biased Random Forest FP Filter + MCL
   - Exp 5: Full Feature Space (X_flagged) vs NMF Latent Space (W_reduced)
5. Computes purity metrics, dominant class labeling, and crosstabs.
6. Outputs a master comparison table and saves it to 'clustering_comparison.csv'.
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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

# ==============================================================================
# CONFIGURATION BLOCK
# ==============================================================================
CONFIG = {
    "random_state": 42,
    "data_path": r"pipeline_cache\ampds_behavior_context_labeled_features.csv",
    "nmf_k": 40,
    "nmf_alpha": 0.4,
    "nmf_max_iter": 3000,
    "pca_dims": 10,
    "kmeans_k_list": [5, 7, 10, 12, 14],
    "mcl_thresholds": [0.3, 0.5, 0.7],
    "mcl_inflations": [1.5, 2.0, 2.5],
    "rf_conf_thresholds": [0.3, 0.5, 0.7],
    "rf_n_estimators": 100,
    "output_csv": "clustering_comparison.csv",
}

# ==============================================================================
# STEP 1: LOAD DATA & RUN BEST NMF PIPELINE TO OBTAIN FLAGGED SAMPLES
# ==============================================================================
def load_and_run_nmf(cfg):
    """Loads dataset, trains best NMF model, and extracts flagged test samples."""
    print("=" * 80)
    print("STEP 1: RUNNING NMF ANOMALY DETECTION PIPELINE")
    print("=" * 80)
    
    df = pd.read_csv(cfg["data_path"]).drop(columns=["window_id"])
    df_reset = df.reset_index(drop=True)
    
    anom_df = df_reset[df_reset["is_anomaly"] == 1]
    norm_df = df_reset[df_reset["is_anomaly"] == 0]
    
    # 70/15/15 Stratified Split
    anom_train, anom_temp = train_test_split(
        anom_df, test_size=0.30, stratify=anom_df["anomaly_type"], random_state=cfg["random_state"]
    )
    anom_val, anom_test = train_test_split(
        anom_temp, test_size=0.50, stratify=anom_temp["anomaly_type"], random_state=cfg["random_state"]
    )
    norm_train, norm_temp = train_test_split(norm_df, test_size=0.30, random_state=cfg["random_state"])
    norm_val, norm_test = train_test_split(norm_temp, test_size=0.50, random_state=cfg["random_state"])
    
    train_df = pd.concat([anom_train, norm_train]).sample(frac=1, random_state=cfg["random_state"]).reset_index(drop=True)
    val_df   = pd.concat([anom_val, norm_val]).sample(frac=1, random_state=cfg["random_state"]).reset_index(drop=True)
    test_df  = pd.concat([anom_test, norm_test]).sample(frac=1, random_state=cfg["random_state"]).reset_index(drop=True)
    
    drop_cols = ["is_anomaly", "anomaly_type"]
    X_train_raw, y_train = train_df.drop(columns=drop_cols), train_df["is_anomaly"]
    X_val_raw, y_val = val_df.drop(columns=drop_cols), val_df["is_anomaly"]
    X_test_raw, y_test = test_df.drop(columns=drop_cols), test_df["is_anomaly"]
    
    # Scale on normal train data only
    scaler = MinMaxScaler()
    X_train_norm_scaled = scaler.fit_transform(X_train_raw[y_train == 0])
    X_val_scaled = np.clip(scaler.transform(X_val_raw), 0, None)
    X_test_scaled = np.clip(scaler.transform(X_test_raw), 0, None)
    
    # Fit NMF model
    nmf = NMF(
        n_components=cfg["nmf_k"], init="nndsvda", solver="cd",
        max_iter=cfg["nmf_max_iter"], tol=1e-3, random_state=cfg["random_state"]
    )
    nmf.fit(X_train_norm_scaled)
    
    # Feature weights: Inverse Normal Train Reconstruction MSE
    W_train = nmf.transform(X_train_norm_scaled)
    X_train_recon = nmf.inverse_transform(W_train)
    mse_per_feat = np.mean((X_train_norm_scaled - X_train_recon) ** 2, axis=0)
    w_norm = (1.0 / (mse_per_feat + 1e-6)) / np.sum(1.0 / (mse_per_feat + 1e-6))
    
    # Scoring helper
    def score_fn(X):
        W = nmf.transform(X)
        X_r = nmf.inverse_transform(W)
        sq_err = (X - X_r) ** 2
        weighted_in_err = np.sqrt(np.sum(sq_err * w_norm, axis=1) * sq_err.shape[1])
        W_r = nmf.transform(X_r)
        latent_err = np.linalg.norm(W - W_r, axis=1)
        return cfg["nmf_alpha"] * weighted_in_err + (1.0 - cfg["nmf_alpha"]) * latent_err
    
    # Validation threshold tuning
    val_scores = score_fn(X_val_scaled)
    thresholds = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 99)))
    best_th, best_f1 = None, -1
    for t in thresholds:
        f = f1_score(y_val, (val_scores >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_th = f, float(t)
            
    test_scores = score_fn(X_test_scaled)
    test_pred = (test_scores >= best_th).astype(int)
    
    # Extract latent representations
    W_val = nmf.transform(X_val_scaled)
    W_test = nmf.transform(X_test_scaled)
    
    # Flagged test samples
    flag_mask = (test_pred == 1)
    W_flagged = W_test[flag_mask]
    X_flagged = X_test_scaled[flag_mask]
    true_types = test_df.loc[flag_mask, "anomaly_type"].values
    is_true_anom = test_df.loc[flag_mask, "is_anomaly"].values
    
    # Flagged validation samples (for training supervised/biased FP filter)
    val_flag_mask = (val_scores >= best_th)
    W_val_flagged = W_val[val_flag_mask]
    y_val_flagged = y_val[val_flag_mask].values
    
    print(f"Total test windows: {len(test_df)}")
    print(f"Flagged anomalies:  {flag_mask.sum()} windows (True Positives: {is_true_anom.sum()}, False Positives: {(is_true_anom == 0).sum()})")
    print(f"Distinct true anomaly types present in flagged set: {len(np.unique(true_types[is_true_anom == 1]))}")
    
    return {
        "W_flagged": W_flagged, "X_flagged": X_flagged,
        "true_types": true_types, "is_true_anom": is_true_anom,
        "W_val_flagged": W_val_flagged, "y_val_flagged": y_val_flagged
    }

# ==============================================================================
# STEP 2: PREPROCESSING (L2 NORMALIZATION + PCA REDUCTION)
# ==============================================================================
def preprocess_latent_vectors(W_flagged, n_components=10, random_state=42):
    """Normalizes latent vectors with L2 norm and reduces dimensions using PCA."""
    W_norm = normalize(W_flagged, norm="l2")
    pca = PCA(n_components=n_components, random_state=random_state)
    W_reduced = pca.fit_transform(W_norm)
    return W_reduced

# ==============================================================================
# HELPER: EVALUATE CLUSTER PURITY & DOMINANT LABELING
# ==============================================================================
def evaluate_cluster_labeling(cluster_labels, true_types, is_true_anom):
    """
    Implements semi-supervised cluster labeling (MTH-IDS Tier 3):
    - Labels each cluster by its dominant anomaly_type
    - Calculates purity overall and strictly on True Positive windows (is_anomaly=1).
    """
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    dominant_type_per_cluster = {}
    purity_per_cluster = {}
    
    tp_correct, tp_total = 0, 0
    all_correct, all_total = 0, len(cluster_labels)
    
    for c in unique_clusters:
        mask_c = (cluster_labels == c)
        types_in_c = true_types[mask_c]
        
        # Dominant type across all samples in cluster
        types_series = pd.Series(types_in_c)
        dom_type = types_series.mode()[0]
        dominant_type_per_cluster[c] = dom_type
        
        # Purity overall
        purity_all = (types_series == dom_type).mean()
        
        # Purity on True Positives only
        tp_mask_c = mask_c & (is_true_anom == 1)
        if tp_mask_c.sum() > 0:
            tp_types_c = pd.Series(true_types[tp_mask_c])
            tp_dom_type = tp_types_c.mode()[0]
            purity_tp = (tp_types_c == tp_dom_type).mean()
            tp_correct += (tp_types_c == tp_dom_type).sum()
            tp_total += len(tp_types_c)
        else:
            purity_tp = 0.0
            
        purity_per_cluster[c] = {
            "dominant_type": dom_type, "purity_all": purity_all,
            "purity_tp": purity_tp, "size": mask_c.sum(), "tp_size": tp_mask_c.sum()
        }
        all_correct += (types_series == dom_type).sum()
        
    mean_purity_tp = (tp_correct / tp_total) if tp_total > 0 else 0.0
    mean_purity_all = (all_correct / all_total) if all_total > 0 else 0.0
    
    assigned_mask = (cluster_labels != -1)
    crosstab = pd.crosstab(
        pd.Series(true_types[assigned_mask], name="True Anomaly Type"),
        pd.Series([dominant_type_per_cluster.get(c, "Unassigned") for c in cluster_labels[assigned_mask]], name="Cluster Dominant Label")
    )
    
    return {
        "n_clusters": len(unique_clusters),
        "mean_purity_tp": mean_purity_tp,
        "mean_purity_all": mean_purity_all,
        "purity_per_cluster": purity_per_cluster,
        "dominant_type_per_cluster": dominant_type_per_cluster,
        "n_assigned": int(assigned_mask.sum()),
        "n_unassigned": int((cluster_labels == -1).sum()),
        "crosstab": crosstab
    }

# ==============================================================================
# EXPERIMENT 1: BASELINE KMEANS + CLUSTER LABELING (MTH-IDS TIER 3)
# ==============================================================================
def run_experiment_1_kmeans(W_reduced, true_types, is_true_anom, k_list=CONFIG["kmeans_k_list"]):
    """Performs KMeans clustering over multiple k values and labels clusters."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: BASELINE KMEANS + CLUSTER LABELING (MTH-IDS Tier 3)")
    print("=" * 80)
    
    best_res, best_k, best_purity = None, None, -1.0
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=10)
        labels = km.fit_predict(W_reduced)
        res = evaluate_cluster_labeling(labels, true_types, is_true_anom)
        print(f"KMeans (k={k:>2}) -> TP Purity: {res['mean_purity_tp']:.4f} | Overall Purity: {res['mean_purity_all']:.4f} | Assigned: {res['n_assigned']}/{len(true_types)}")
        if res["mean_purity_tp"] > best_purity:
            best_purity, best_k, best_res = res["mean_purity_tp"], k, res
            
    print(f"--> Best KMeans: k={best_k} with TP Purity = {best_purity:.4f}")
    best_res["best_param"] = best_k
    return best_res

# ==============================================================================
# EXPERIMENT 2: GRAPH-BASED MCL + CLUSTER LABELING
# ==============================================================================
def run_experiment_2_mcl(W_reduced, true_types, is_true_anom, thresholds=CONFIG["mcl_thresholds"], inflations=CONFIG["mcl_inflations"]):
    """Builds cosine similarity graph and performs Markov Clustering (MCL)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: GRAPH-BASED MCL + CLUSTER LABELING")
    print("=" * 80)
    
    sim_matrix = cosine_similarity(W_reduced)
    best_res, best_params, best_purity = None, None, -1.0
    
    for thresh in thresholds:
        adj = np.copy(sim_matrix)
        adj[adj < thresh] = 0.0
        np.fill_diagonal(adj, 1.0)
        
        for inf in inflations:
            result = mc.run_mcl(adj, inflation=inf)
            clusters = mc.get_clusters(result)
            
            labels = np.full(len(W_reduced), -1, dtype=int)
            for c_id, cluster_nodes in enumerate(clusters):
                for node in cluster_nodes:
                    labels[node] = c_id
                    
            res = evaluate_cluster_labeling(labels, true_types, is_true_anom)
            print(f"MCL (thresh={thresh:.1f}, inflation={inf:.1f}) -> Clusters: {res['n_clusters']:>2} | TP Purity: {res['mean_purity_tp']:.4f} | Overall Purity: {res['mean_purity_all']:.4f}")
            if res["mean_purity_tp"] > best_purity:
                best_purity, best_params, best_res = res["mean_purity_tp"], (thresh, inf), res
                
    print(f"--> Best MCL: threshold={best_params[0]}, inflation={best_params[1]} -> TP Purity = {best_purity:.4f} ({best_res['n_clusters']} clusters)")
    best_res["best_param"] = best_params
    return best_res

# ==============================================================================
# EXPERIMENT 3: FP FILTER + KMEANS (MTH-IDS TIER 4 BIASED CLASSIFIER)
# ==============================================================================
def run_experiment_3_fp_kmeans(data_dict, best_k_exp1, conf_thresholds=CONFIG["rf_conf_thresholds"]):
    """Trains Random Forest biased classifier to filter FPs, then runs KMeans on survivors."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: FP FILTER + KMEANS (MTH-IDS Tier 4 Biased Classifier)")
    print("=" * 80)
    
    W_val_flagged = data_dict["W_val_flagged"]
    y_val_flagged = data_dict["y_val_flagged"]
    W_test_flagged = data_dict["W_flagged"]
    true_types = data_dict["true_types"]
    is_true_anom = data_dict["is_true_anom"]
    
    # Train Random Forest FP/TP Biased Classifier on validation flagged anomalies
    rf = RandomForestClassifier(n_estimators=CONFIG["rf_n_estimators"], random_state=CONFIG["random_state"])
    rf.fit(W_val_flagged, y_val_flagged)
    p_true_anom = rf.predict_proba(W_test_flagged)[:, 1] if len(rf.classes_) > 1 else np.ones(len(W_test_flagged))
    
    best_res, best_th, best_purity = None, None, -1.0
    for th in conf_thresholds:
        survivor_mask = (p_true_anom >= th)
        if survivor_mask.sum() < best_k_exp1:
            continue
            
        W_survivors = preprocess_latent_vectors(W_test_flagged[survivor_mask], n_components=min(CONFIG["pca_dims"], survivor_mask.sum()-1))
        km = KMeans(n_clusters=best_k_exp1, random_state=CONFIG["random_state"], n_init=10)
        labels_survivors = km.fit_predict(W_survivors)
        
        # Map labels back to full flagged test set (-1 for filtered FPs)
        full_labels = np.full(len(W_test_flagged), -1, dtype=int)
        full_labels[survivor_mask] = labels_survivors
        
        res = evaluate_cluster_labeling(full_labels, true_types, is_true_anom)
        tp_kept = (survivor_mask & (is_true_anom == 1)).sum()
        fp_removed = (~survivor_mask & (is_true_anom == 0)).sum()
        print(f"FP+KMeans (P_th={th:.1f}) -> Kept {survivor_mask.sum():>2}/{len(W_test_flagged)} (TPs kept: {tp_kept}/{is_true_anom.sum()}, FPs removed: {fp_removed}/{(is_true_anom==0).sum()}) | TP Purity: {res['mean_purity_tp']:.4f}")
        
        if res["mean_purity_tp"] > best_purity:
            best_purity, best_th, best_res = res["mean_purity_tp"], th, res
            best_res["tp_kept"] = tp_kept
            best_res["fp_removed"] = fp_removed
            
    print(f"--> Best FP+KMeans: P_th={best_th} -> TP Purity = {best_purity:.4f}")
    best_res["best_param"] = best_th
    return best_res, rf

# ==============================================================================
# EXPERIMENT 4: FP FILTER + MCL
# ==============================================================================
def run_experiment_4_fp_mcl(data_dict, rf_model, best_mcl_params, best_rf_th):
    """Combines Random Forest FP Filter with Markov Clustering."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: FP FILTER + MCL (Combining Biased Classifier & Graph MCL)")
    print("=" * 80)
    
    W_test_flagged = data_dict["W_flagged"]
    true_types = data_dict["true_types"]
    is_true_anom = data_dict["is_true_anom"]
    thresh, inf = best_mcl_params
    
    p_true_anom = rf_model.predict_proba(W_test_flagged)[:, 1] if len(rf_model.classes_) > 1 else np.ones(len(W_test_flagged))
    survivor_mask = (p_true_anom >= best_rf_th)
    
    W_survivors = preprocess_latent_vectors(W_test_flagged[survivor_mask], n_components=min(CONFIG["pca_dims"], survivor_mask.sum()-1))
    sim_matrix = cosine_similarity(W_survivors)
    sim_matrix[sim_matrix < thresh] = 0.0
    np.fill_diagonal(sim_matrix, 1.0)
    
    result = mc.run_mcl(sim_matrix, inflation=inf)
    clusters = mc.get_clusters(result)
    
    labels_survivors = np.full(survivor_mask.sum(), -1, dtype=int)
    for c_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            labels_survivors[node] = c_id
            
    full_labels = np.full(len(W_test_flagged), -1, dtype=int)
    full_labels[survivor_mask] = labels_survivors
    
    res = evaluate_cluster_labeling(full_labels, true_types, is_true_anom)
    tp_kept = (survivor_mask & (is_true_anom == 1)).sum()
    fp_removed = (~survivor_mask & (is_true_anom == 0)).sum()
    res["tp_kept"] = tp_kept
    res["fp_removed"] = fp_removed
    res["best_param"] = (best_rf_th, thresh, inf)
    
    print(f"FP+MCL (P_th={best_rf_th:.1f}, thresh={thresh:.1f}, inf={inf:.1f}) -> Clusters: {res['n_clusters']} | Kept {survivor_mask.sum()}/{len(W_test_flagged)} (TPs kept: {tp_kept}, FPs removed: {fp_removed}) | TP Purity: {res['mean_purity_tp']:.4f}")
    return res

# ==============================================================================
# EXPERIMENT 5: FULL FEATURE SPACE (X_flagged) VS LATENT SPACE (W_reduced)
# ==============================================================================
def run_experiment_5_raw_features(data_dict, best_exp_name, exp1_k, best_mcl_params):
    """Tests whether clustering directly on full feature space (X_flagged) beats NMF latent space."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: RAW FEATURE SPACE (X_flagged) VS LATENT SPACE (W_reduced)")
    print("=" * 80)
    
    X_flagged = data_dict["X_flagged"]
    true_types = data_dict["true_types"]
    is_true_anom = data_dict["is_true_anom"]
    
    # Normalize and PCA-reduce raw 830 features to 10 dims
    X_norm = normalize(X_flagged, norm="l2")
    X_reduced = PCA(n_components=CONFIG["pca_dims"], random_state=CONFIG["random_state"]).fit_transform(X_norm)
    
    # Run KMeans baseline on raw features
    km = KMeans(n_clusters=exp1_k, random_state=CONFIG["random_state"], n_init=10)
    km_labels = km.fit_predict(X_reduced)
    res_km = evaluate_cluster_labeling(km_labels, true_types, is_true_anom)
    res_km["best_param"] = f"KMeans k={exp1_k} (Raw Features)"
    print(f"Raw Features + KMeans (k={exp1_k}) -> TP Purity: {res_km['mean_purity_tp']:.4f} | Overall Purity: {res_km['mean_purity_all']:.4f}")
    
    # Run MCL on raw features
    thresh, inf = best_mcl_params
    sim_matrix = cosine_similarity(X_reduced)
    sim_matrix[sim_matrix < thresh] = 0.0
    np.fill_diagonal(sim_matrix, 1.0)
    result = mc.run_mcl(sim_matrix, inflation=inf)
    clusters = mc.get_clusters(result)
    mcl_labels = np.full(len(X_reduced), -1, dtype=int)
    for c_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            mcl_labels[node] = c_id
    res_mcl = evaluate_cluster_labeling(mcl_labels, true_types, is_true_anom)
    res_mcl["best_param"] = f"MCL t={thresh} i={inf} (Raw Features)"
    print(f"Raw Features + MCL (thresh={thresh}, inf={inf}) -> Clusters: {res_mcl['n_clusters']} | TP Purity: {res_mcl['mean_purity_tp']:.4f} | Overall Purity: {res_mcl['mean_purity_all']:.4f}")
    
    return res_km if res_km["mean_purity_tp"] >= res_mcl["mean_purity_tp"] else res_mcl

# ==============================================================================
# MAIN EXECUTION & MASTER REPORTING
# ==============================================================================
def main():
    start_time = time.perf_counter()
    
    # 1. Pipeline extraction
    data_dict = load_and_run_nmf(CONFIG)
    W_reduced = preprocess_latent_vectors(data_dict["W_flagged"], n_components=CONFIG["pca_dims"], random_state=CONFIG["random_state"])
    
    # 2. Run Experiments
    exp1_res = run_experiment_1_kmeans(W_reduced, data_dict["true_types"], data_dict["is_true_anom"])
    exp2_res = run_experiment_2_mcl(W_reduced, data_dict["true_types"], data_dict["is_true_anom"])
    exp3_res, rf_model = run_experiment_3_fp_kmeans(data_dict, exp1_res["best_param"])
    exp4_res = run_experiment_4_fp_mcl(data_dict, rf_model, exp2_res["best_param"], exp3_res["best_param"])
    exp5_res = run_experiment_5_raw_features(data_dict, "Best", exp1_res["best_param"], exp2_res["best_param"])
    
    # 3. Master Comparison Table
    experiments_summary = [
        {"experiment": f"KMeans (k={exp1_res['best_param']})", "n_clusters": exp1_res["n_clusters"], "mean_purity_tp": round(exp1_res["mean_purity_tp"], 4), "n_assigned": f"{exp1_res['n_assigned']}/{len(data_dict['true_types'])}", "notes": f"MTH-IDS Tier 3 baseline on W_reduced"},
        {"experiment": f"MCL (t={exp2_res['best_param'][0]}, i={exp2_res['best_param'][1]})", "n_clusters": exp2_res["n_clusters"], "mean_purity_tp": round(exp2_res["mean_purity_tp"], 4), "n_assigned": f"{exp2_res['n_assigned']}/{len(data_dict['true_types'])}", "notes": f"Graph-based Markov clustering on W_reduced"},
        {"experiment": f"FP+KMeans (P_th={exp3_res['best_param']})", "n_clusters": exp3_res["n_clusters"], "mean_purity_tp": round(exp3_res["mean_purity_tp"], 4), "n_assigned": f"{exp3_res['n_assigned']}/{len(data_dict['true_types'])}", "notes": f"Tier 4 RF filter (removed {exp3_res['fp_removed']} FPs, kept {exp3_res['tp_kept']} TPs)"},
        {"experiment": f"FP+MCL (P_th={exp4_res['best_param'][0]}, t={exp4_res['best_param'][1]}, i={exp4_res['best_param'][2]})", "n_clusters": exp4_res["n_clusters"], "mean_purity_tp": round(exp4_res["mean_purity_tp"], 4), "n_assigned": f"{exp4_res['n_assigned']}/{len(data_dict['true_types'])}", "notes": f"RF filter + Graph MCL (removed {exp4_res['fp_removed']} FPs, kept {exp4_res['tp_kept']} TPs)"},
        {"experiment": f"Raw Features ({exp5_res['best_param']})", "n_clusters": exp5_res["n_clusters"], "mean_purity_tp": round(exp5_res["mean_purity_tp"], 4), "n_assigned": f"{exp5_res['n_assigned']}/{len(data_dict['true_types'])}", "notes": f"Direct on 830 raw features vs W_reduced"}
    ]
    
    summary_df = pd.DataFrame(experiments_summary)
    summary_df.to_csv(CONFIG["output_csv"], index=False)
    
    print("\n" + "=" * 80)
    print("MASTER EXPERIMENT COMPARISON TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison summary to: {CONFIG['output_csv']}")
    
    # 4. Detailed Report for the Winning Approach
    all_experiments = [
        ("KMeans (W_reduced)", exp1_res),
        ("MCL (W_reduced)", exp2_res),
        ("FP Filter + KMeans", exp3_res),
        ("FP Filter + MCL", exp4_res),
        ("Raw Features", exp5_res)
    ]
    winner_name, winner_res = max(all_experiments, key=lambda x: x[1]["mean_purity_tp"])
    
    print("\n" + "=" * 80)
    print(f"SINGLE BEST APPROACH: {winner_name} (TP Purity = {winner_res['mean_purity_tp']:.4f})")
    print("=" * 80)
    
    print("\nCluster Labeling Breakdown (Cluster Number -> Dominant Anomaly Type):")
    cluster_details = []
    for c_id, info in winner_res["purity_per_cluster"].items():
        cluster_details.append({
            "Cluster_ID": f"Cluster {c_id}",
            "Dominant_Label": info["dominant_type"],
            "Cluster_Size": info["size"],
            "TP_Size": info["tp_size"],
            "TP_Purity": f"{info['purity_tp']*100:.1f}%",
            "Overall_Purity": f"{info['purity_all']*100:.1f}%"
        })
    print(pd.DataFrame(cluster_details).to_string(index=False))
    
    print("\nFull Crosstab (True Anomaly Type vs Predicted Cluster Label):")
    print(winner_res["crosstab"].to_string())
    
    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 80)
    print(f"FRAMEWORK EXECUTION COMPLETED IN {total_time:.2f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()

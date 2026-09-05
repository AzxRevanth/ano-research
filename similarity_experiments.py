"""
================================================================================
EXPERIMENT SUITE: SIMILARITY METRIC BENCHMARK FOR GRAPH CLUSTERING (MCL)
================================================================================
Compares multiple similarity / affinity metrics for graph-based clustering:
  1. Cosine Similarity (baseline)
  2. Spearman Rank Correlation (non-linear monotonic rank matching)
  3. Pearson Correlation (linear profile alignment)
  4. RBF / Gaussian Kernel (Euclidean distance with median heuristic gamma)
  5. Manhattan / Exponential Laplacian Kernel (L1 cityblock distance)

Evaluates both:
  - Macro-average TP Purity (unweighted)
  - Size-Weighted Micro TP Purity
  - Clusters with >= 3 members (robustness against singleton inflation)
  - Full flagged set (N=87) and FP-filtered set (N=35)
================================================================================
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import markov_clustering as mc

from cluster_labeling_framework import load_and_run_nmf, preprocess_latent_vectors, CONFIG

def compute_similarity_matrices(X):
    """Computes various pairwise similarity matrices."""
    matrices = {}
    
    # 1. Cosine
    matrices["Cosine"] = cosine_similarity(X)
    
    # 2. Pearson Correlation
    # np.corrcoef computes correlation along rows when rowvar=True (default)
    matrices["Pearson"] = np.corrcoef(X)
    
    # 3. Spearman Rank Correlation
    rho, _ = spearmanr(X, axis=1)
    matrices["Spearman"] = rho
    
    # 4. RBF / Gaussian Kernel on Euclidean Distance
    euc_dist = euclidean_distances(X)
    med_euc = np.median(euc_dist)
    gamma_rbf = 1.0 / (2.0 * (med_euc ** 2) + 1e-6)
    matrices["RBF_Gaussian"] = np.exp(-gamma_rbf * (euc_dist ** 2))
    
    # 5. Manhattan Laplacian Kernel (Exponential of L1)
    man_dist = squareform(pdist(X, metric="cityblock"))
    med_man = np.median(man_dist)
    gamma_man = 1.0 / (med_man + 1e-6)
    matrices["Manhattan_Laplacian"] = np.exp(-gamma_man * man_dist)
    
    return matrices

def evaluate_mcl(sim_matrix, true_types, is_true_anom, thresh, inf):
    """Runs MCL on thresholded similarity matrix and computes metrics."""
    adj = np.copy(sim_matrix)
    # Clip negative correlations if any (e.g., in Pearson/Spearman)
    adj = np.clip(adj, 0.0, 1.0)
    adj[adj < thresh] = 0.0
    np.fill_diagonal(adj, 1.0)
    
    # If all non-diagonal entries are 0, return disconnected
    if np.sum(adj > 0) <= len(adj):
        return None
        
    try:
        result = mc.run_mcl(adj, inflation=inf)
        clusters = mc.get_clusters(result)
    except Exception:
        return None
        
    n_clusters = len(clusters)
    labels = np.full(len(sim_matrix), -1, dtype=int)
    for c_id, nodes in enumerate(clusters):
        for n in nodes:
            labels[n] = c_id
            
    assigned_mask = (labels != -1)
    labels_a = labels[assigned_mask]
    types_a = true_types[assigned_mask]
    is_tp_a = is_true_anom[assigned_mask]
    
    tp_hits, tp_total = 0, 0
    all_hits, all_total = 0, len(labels_a)
    
    cluster_stats = []
    for c in np.unique(labels_a):
        mask_c = (labels_a == c)
        types_c = types_a[mask_c]
        is_tp_c = is_tp_a[mask_c]
        size = len(types_c)
        tp_size = is_tp_c.sum()
        
        dom_all = pd.Series(types_c).mode()[0]
        purity_all = (types_c == dom_all).mean()
        all_hits += (types_c == dom_all).sum()
        
        if tp_size > 0:
            tp_types_c = types_c[is_tp_c == 1]
            dom_tp = pd.Series(tp_types_c).mode()[0]
            purity_tp = (tp_types_c == dom_tp).mean()
            tp_hits += (tp_types_c == dom_tp).sum()
            tp_total += tp_size
        else:
            purity_tp = 0.0
            dom_tp = "None"
            
        cluster_stats.append({
            "cluster": c, "size": size, "tp_size": tp_size,
            "dom_tp": dom_tp, "purity_tp": purity_tp,
            "dom_all": dom_all, "purity_all": purity_all
        })
        
    df_c = pd.DataFrame(cluster_stats)
    macro_tp = df_c[df_c["tp_size"] > 0]["purity_tp"].mean() if len(df_c[df_c["tp_size"] > 0]) > 0 else 0
    micro_tp = (tp_hits / tp_total) if tp_total > 0 else 0
    
    n_singletons = (df_c["size"] == 1).sum()
    df_ge_3 = df_c[df_c["size"] >= 3]
    n_ge_3 = len(df_ge_3)
    
    if len(df_ge_3) > 0 and df_ge_3["tp_size"].sum() > 0:
        micro_tp_ge_3 = (df_ge_3["purity_tp"] * df_ge_3["tp_size"]).sum() / df_ge_3["tp_size"].sum()
    else:
        micro_tp_ge_3 = 0.0
        
    return {
        "n_clusters": n_clusters,
        "macro_tp_purity": macro_tp,
        "weighted_tp_purity": micro_tp,
        "micro_tp_ge_3": micro_tp_ge_3,
        "n_ge_3": n_ge_3,
        "n_singletons": n_singletons,
        "cluster_df": df_c
    }

def run_similarity_suite():
    print("=" * 80)
    print("RUNNING SIMILARITY SUITE BENCHMARK")
    print("=" * 80)
    
    data = load_and_run_nmf(CONFIG)
    W_flagged = data["W_flagged"]
    true_types = data["true_types"]
    is_true_anom = data["is_true_anom"]
    
    # Preprocess W vectors (L2 norm + PCA 10)
    W_reduced = preprocess_latent_vectors(W_flagged, n_components=CONFIG["pca_dims"], random_state=CONFIG["random_state"])
    
    # Prepare FP Filtered subset (P >= 0.70)
    rf = RandomForestClassifier(n_estimators=CONFIG["rf_n_estimators"], random_state=CONFIG["random_state"])
    rf.fit(data["W_val_flagged"], data["y_val_flagged"])
    p_anom = rf.predict_proba(W_flagged)[:, 1] if len(rf.classes_) > 1 else np.ones(len(W_flagged))
    surv = (p_anom >= 0.70)
    
    W_surv_red = preprocess_latent_vectors(W_flagged[surv], n_components=min(CONFIG["pca_dims"], surv.sum() - 1), random_state=CONFIG["random_state"])
    true_types_surv = true_types[surv]
    is_true_anom_surv = is_true_anom[surv]
    
    # -------------------------------------------------------------
    # PART 1: Benchmark on all 87 flagged alarms (No filter)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: SIMILARITY BENCHMARK ON ALL FLAGGED ALARMS (N=87)")
    print("=" * 80)
    
    sim_matrices_87 = compute_similarity_matrices(W_reduced)
    thresh_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    inf_list = [1.5, 2.0, 2.5]
    
    results_87 = []
    best_clusters_87 = {}
    
    for sim_name, sim_mat in sim_matrices_87.items():
        best_for_sim = None
        best_weighted = -1.0
        
        for t in thresh_list:
            for inf in inf_list:
                eval_res = evaluate_mcl(sim_mat, true_types, is_true_anom, t, inf)
                if eval_res is None or eval_res["n_clusters"] <= 1 or eval_res["n_clusters"] >= 50:
                    continue
                
                # Selection based on weighted TP purity
                if eval_res["weighted_tp_purity"] > best_weighted:
                    best_weighted = eval_res["weighted_tp_purity"]
                    best_for_sim = {
                        "Metric": sim_name,
                        "Threshold": t,
                        "Inflation": inf,
                        "N_Clusters": eval_res["n_clusters"],
                        "Singletons": eval_res["n_singletons"],
                        "Clusters_ge_3": eval_res["n_ge_3"],
                        "Weighted_TP_Purity": eval_res["weighted_tp_purity"],
                        "Purity_ge_3": eval_res["micro_tp_ge_3"],
                        "Macro_TP_Purity": eval_res["macro_tp_purity"],
                    }
                    best_clusters_87[sim_name] = eval_res["cluster_df"]
                    
        if best_for_sim:
            results_87.append(best_for_sim)
            print(f"[{sim_name:20s}] Best t={best_for_sim['Threshold']}, i={best_for_sim['Inflation']} -> "
                  f"Weighted TP Purity: {best_for_sim['Weighted_TP_Purity']:.4f} | "
                  f"Purity (>=3): {best_for_sim['Purity_ge_3']:.4f} | "
                  f"Clusters: {best_for_sim['N_Clusters']} (>=3: {best_for_sim['Clusters_ge_3']}, 1s: {best_for_sim['Singletons']})")

    df_res_87 = pd.DataFrame(results_87).sort_values("Weighted_TP_Purity", ascending=False)
    print("\n--- SUMMARY RANKING (N=87 FLAGGED WINDOWS) ---")
    print(df_res_87.to_string(index=False))

    # -------------------------------------------------------------
    # PART 2: Benchmark on FP-Filtered Alarms (N=35)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: SIMILARITY BENCHMARK ON FP-FILTERED ALARMS (N=35)")
    print("=" * 80)
    
    sim_matrices_35 = compute_similarity_matrices(W_surv_red)
    results_35 = []
    best_clusters_35 = {}
    
    for sim_name, sim_mat in sim_matrices_35.items():
        best_for_sim = None
        best_weighted = -1.0
        
        for t in thresh_list:
            for inf in inf_list:
                eval_res = evaluate_mcl(sim_mat, true_types_surv, is_true_anom_surv, t, inf)
                if eval_res is None or eval_res["n_clusters"] <= 1 or eval_res["n_clusters"] >= 30:
                    continue
                
                if eval_res["weighted_tp_purity"] > best_weighted:
                    best_weighted = eval_res["weighted_tp_purity"]
                    best_for_sim = {
                        "Metric": sim_name,
                        "Threshold": t,
                        "Inflation": inf,
                        "N_Clusters": eval_res["n_clusters"],
                        "Singletons": eval_res["n_singletons"],
                        "Clusters_ge_3": eval_res["n_ge_3"],
                        "Weighted_TP_Purity": eval_res["weighted_tp_purity"],
                        "Purity_ge_3": eval_res["micro_tp_ge_3"],
                        "Macro_TP_Purity": eval_res["macro_tp_purity"],
                    }
                    best_clusters_35[sim_name] = eval_res["cluster_df"]
                    
        if best_for_sim:
            results_35.append(best_for_sim)
            print(f"[{sim_name:20s}] Best t={best_for_sim['Threshold']}, i={best_for_sim['Inflation']} -> "
                  f"Weighted TP Purity: {best_for_sim['Weighted_TP_Purity']:.4f} | "
                  f"Purity (>=3): {best_for_sim['Purity_ge_3']:.4f} | "
                  f"Clusters: {best_for_sim['N_Clusters']} (>=3: {best_for_sim['Clusters_ge_3']}, 1s: {best_for_sim['Singletons']})")

    df_res_35 = pd.DataFrame(results_35).sort_values("Weighted_TP_Purity", ascending=False)
    print("\n--- SUMMARY RANKING (N=35 FP-FILTERED WINDOWS) ---")
    print(df_res_35.to_string(index=False))
    
    # Save combined results
    df_res_87["Dataset"] = "All Flagged (N=87)"
    df_res_35["Dataset"] = "FP Filtered (N=35)"
    master_sim_df = pd.concat([df_res_87, df_res_35]).reset_index(drop=True)
    master_sim_df.to_csv("similarity_metrics_comparison.csv", index=False)
    print("\nSaved full similarity comparison to 'similarity_metrics_comparison.csv'")
    
    # Detail on the top performers
    print("\n" + "=" * 80)
    print("DETAILED CLUSTER COMPOSITION OF TOP SIMILARITY METRIC ON N=87")
    print("=" * 80)
    top_metric_87 = df_res_87.iloc[0]["Metric"]
    print(f"Top Metric on N=87: {top_metric_87}")
    print(best_clusters_87[top_metric_87].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("DETAILED CLUSTER COMPOSITION OF TOP SIMILARITY METRIC ON N=35")
    print("=" * 80)
    top_metric_35 = df_res_35.iloc[0]["Metric"]
    print(f"Top Metric on N=35: {top_metric_35}")
    print(best_clusters_35[top_metric_35].to_string(index=False))

if __name__ == "__main__":
    run_similarity_suite()

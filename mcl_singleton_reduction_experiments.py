"""
================================================================================
MCL EXPERIMENT SUITE: SINGLETON REDUCTION & PURITY PRESERVATION
================================================================================
Investigates:
  Exp 1: Edge-pruning threshold sweep (0.60, 0.65, 0.70, 0.75, 0.80) @ PCA=10, inf=1.5
  Exp 2: MCL inflation sweep (1.2, 1.3, 1.5, 1.7, 2.0) @ PCA=10, thresh=0.80
  Exp 3: Combined grid (thresholds: 0.65, 0.70, 0.75, 0.80; inflations: 1.2, 1.3, 1.5) @ PCA=10
  Exp 4: PCA dimensions sweep (5, 10, 15, 20) tested across baseline & grid settings

Outputs comprehensive metrics:
  - Threshold, Inflation, PCA_dims
  - N_clusters, Singletons, Clusters_ge_3, Largest_cluster_size
  - Size-weighted TP purity, Purity_ge_3
  - ARI, NMI, FMI (evaluated on TP windows)
================================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    homogeneity_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)
import markov_clustering as mc

from MCL_testing import load_and_split_data, train_weighted_nmf, CONFIG

def run_mcl_eval(W_surv, true_types_surv, is_true_anom_surv, pca_dim, thresh, inf, min_size=3):
    """
    Runs L2 norm -> PCA -> Spearman correlation -> threshold pruning -> MCL.
    Computes all requested metrics.
    """
    n_samples = len(W_surv)
    
    # 1. Representation
    W_norm = normalize(W_surv, norm="l2")
    pca = PCA(n_components=pca_dim, random_state=CONFIG["random_state"])
    W_reduced = pca.fit_transform(W_norm)
    
    # 2. Spearman Correlation & Graph Construction
    rho_matrix, _ = spearmanr(W_reduced, axis=1)
    adj = np.copy(rho_matrix)
    adj = np.clip(adj, 0.0, 1.0)
    adj[adj < thresh] = 0.0
    np.fill_diagonal(adj, 1.0)
    
    # Check connectivity
    if np.sum(adj > 0) <= n_samples:
        # All off-diagonal pruned -> every node is isolated
        clusters = [[i] for i in range(n_samples)]
    else:
        try:
            mcl_res = mc.run_mcl(adj, inflation=inf)
            clusters = mc.get_clusters(mcl_res)
        except Exception:
            clusters = [[i] for i in range(n_samples)]
            
    labels = np.full(n_samples, -1, dtype=int)
    for c_id, nodes in enumerate(clusters):
        for n in nodes:
            labels[n] = c_id
            
    # Slices for True Positives
    tp_mask = (is_true_anom_surv == 1)
    y_true_tp = true_types_surv[tp_mask]
    clusters_tp = labels[tp_mask]
    
    # Purity calculations
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    cluster_stats = []
    tp_correct, tp_total = 0, 0
    
    for c in unique_clusters:
        mask_c = (labels == c)
        types_c = pd.Series(true_types_surv[mask_c])
        size = mask_c.sum()
        
        tp_mask_c = mask_c & tp_mask
        tp_size = tp_mask_c.sum()
        
        if tp_size > 0:
            tp_types_c = pd.Series(true_types_surv[tp_mask_c])
            dom_tp = tp_types_c.mode()[0]
            purity_tp = (tp_types_c == dom_tp).mean()
            tp_hits = (tp_types_c == dom_tp).sum()
            tp_correct += tp_hits
            tp_total += tp_size
        else:
            purity_tp = 0.0
            tp_hits = 0
            dom_tp = "None"
            
        cluster_stats.append({
            "cluster": c,
            "size": size,
            "tp_size": tp_size,
            "tp_hits": tp_hits,
            "purity_tp": purity_tp,
        })
        
    df_c = pd.DataFrame(cluster_stats)
    
    # Metrics
    n_singletons = (df_c["size"] == 1).sum()
    df_ge_3 = df_c[df_c["size"] >= min_size]
    n_ge_3 = len(df_ge_3)
    largest_cluster_size = df_c["size"].max()
    
    weighted_tp_purity = tp_correct / tp_total if tp_total > 0 else 0.0
    
    if n_ge_3 > 0 and df_ge_3["tp_size"].sum() > 0:
        purity_ge_3 = df_ge_3["tp_hits"].sum() / df_ge_3["tp_size"].sum()
    else:
        purity_ge_3 = 0.0
        
    ari = adjusted_rand_score(y_true_tp, clusters_tp)
    nmi = normalized_mutual_info_score(y_true_tp, clusters_tp)
    fmi = fowlkes_mallows_score(y_true_tp, clusters_tp)
    homo = homogeneity_score(y_true_tp, clusters_tp)
    
    return {
        "Threshold": thresh,
        "Inflation": inf,
        "PCA_dims": pca_dim,
        "N_clusters": n_clusters,
        "Singletons": n_singletons,
        "Clusters_ge_3": n_ge_3,
        "Largest_cluster_size": largest_cluster_size,
        "Weighted_TP_purity": weighted_tp_purity,
        "Purity_ge_3": purity_ge_3,
        "ARI": ari,
        "NMI": nmi,
        "FMI": fmi,
        "Homogeneity": homo,
    }

def main():
    print("=" * 80)
    print("STARTING SYSTEMATIC MCL HYPERPARAMETER EXPERIMENTS")
    print("=" * 80)
    
    # 1. Pipeline extraction
    data = load_and_split_data(CONFIG["data_path"], random_state=CONFIG["random_state"])
    nmf_data = train_weighted_nmf(data, CONFIG)
    
    W_val_flagged = nmf_data["W_val_flagged"]
    y_val_flagged = nmf_data["y_val_flagged"]
    W_test_flagged = nmf_data["W_flagged"]
    true_types = nmf_data["true_types"]
    is_true_anom = nmf_data["is_true_anom"]
    
    # Tier 4 RF filter (P >= 0.70)
    rf = RandomForestClassifier(n_estimators=CONFIG["rf_n_estimators"], random_state=CONFIG["random_state"])
    rf.fit(W_val_flagged, y_val_flagged)
    probs = rf.predict_proba(W_test_flagged)[:, 1] if len(rf.classes_) > 1 else np.ones(len(W_test_flagged))
    surv_mask = (probs >= CONFIG["rf_conf_threshold"])
    
    W_surv = W_test_flagged[surv_mask]
    true_types_surv = true_types[surv_mask]
    is_true_anom_surv = is_true_anom[surv_mask]
    
    print(f"Surviving alarms: {len(W_surv)} (TP: {(is_true_anom_surv == 1).sum()}, FP: {(is_true_anom_surv == 0).sum()})")
    
    records = []
    
    # --------------------------------------------------------------------------
    # Experiment 1: Edge-pruning threshold sweep (PCA=10, inf=1.5)
    # --------------------------------------------------------------------------
    print("\n--- Running Experiment 1: Edge-pruning Threshold Sweep ---")
    thresholds_exp1 = [0.60, 0.65, 0.70, 0.75, 0.80]
    for t in thresholds_exp1:
        res = run_mcl_eval(W_surv, true_types_surv, is_true_anom_surv, pca_dim=10, thresh=t, inf=1.5)
        res["Experiment"] = "Exp 1: Threshold"
        records.append(res)
        
    # --------------------------------------------------------------------------
    # Experiment 2: MCL inflation sweep (PCA=10, thresh=0.80)
    # --------------------------------------------------------------------------
    print("--- Running Experiment 2: MCL Inflation Sweep ---")
    inflations_exp2 = [1.2, 1.3, 1.5, 1.7, 2.0]
    for inf in inflations_exp2:
        # If t=0.80, inf=1.5 already computed in Exp 1, we still compute or avoid duplicate
        res = run_mcl_eval(W_surv, true_types_surv, is_true_anom_surv, pca_dim=10, thresh=0.80, inf=inf)
        res["Experiment"] = "Exp 2: Inflation"
        records.append(res)
        
    # --------------------------------------------------------------------------
    # Experiment 3: Combined threshold + inflation grid (PCA=10)
    # --------------------------------------------------------------------------
    print("--- Running Experiment 3: Combined Threshold + Inflation Grid ---")
    thresholds_exp3 = [0.65, 0.70, 0.75, 0.80]
    inflations_exp3 = [1.2, 1.3, 1.5]
    for t in thresholds_exp3:
        for inf in inflations_exp3:
            res = run_mcl_eval(W_surv, true_types_surv, is_true_anom_surv, pca_dim=10, thresh=t, inf=inf)
            res["Experiment"] = "Exp 3: Threshold+Inflation"
            records.append(res)
            
    # --------------------------------------------------------------------------
    # Experiment 4: PCA dimensions sweep (5, 10, 15, 20)
    # --------------------------------------------------------------------------
    print("--- Running Experiment 4: PCA Dimensions Sweep ---")
    pca_dims = [5, 10, 15, 20]
    # Test on baseline (t=0.80, inf=1.5) and top threshold candidates (t=0.70, 0.75)
    for dim in pca_dims:
        for t, inf in [(0.80, 1.5), (0.75, 1.5), (0.70, 1.3), (0.65, 1.3)]:
            res = run_mcl_eval(W_surv, true_types_surv, is_true_anom_surv, pca_dim=dim, thresh=t, inf=inf)
            res["Experiment"] = f"Exp 4: PCA Dim {dim}"
            records.append(res)

    # Convert to DataFrame
    all_df = pd.DataFrame(records)
    
    # Drop exact duplicates if any across experiment definitions
    all_df_unique = all_df.drop_duplicates(subset=["Threshold", "Inflation", "PCA_dims"]).copy()
    
    # Save CSV
    output_csv = "mcl_singleton_reduction_results.csv"
    all_df_unique.to_csv(output_csv, index=False)
    print(f"\nSaved all {len(all_df_unique)} unique configurations to: {output_csv}")
    
    # Display baseline
    baseline = all_df_unique[(all_df_unique["Threshold"] == 0.80) & (all_df_unique["Inflation"] == 1.5) & (all_df_unique["PCA_dims"] == 10)].iloc[0]
    print("\n" + "=" * 80)
    print("CURRENT BASELINE CONFIGURATION")
    print("=" * 80)
    print(f"Threshold: {baseline['Threshold']}, Inflation: {baseline['Inflation']}, PCA_dims: {baseline['PCA_dims']}")
    print(f"Clusters: {baseline['N_clusters']}, Singletons: {baseline['Singletons']}, Clusters >= 3: {baseline['Clusters_ge_3']}, Largest Size: {baseline['Largest_cluster_size']}")
    print(f"Weighted TP Purity: {baseline['Weighted_TP_purity']*100:.2f}%, Purity (>=3): {baseline['Purity_ge_3']*100:.2f}%")
    print(f"ARI: {baseline['ARI']:.4f}, NMI: {baseline['NMI']:.4f}, FMI: {baseline['FMI']:.4f}")
    
    # Print sorted tables
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 RESULTS (Edge-pruning Threshold @ PCA=10, inf=1.5)")
    print("=" * 80)
    exp1_df = all_df_unique[(all_df_unique["PCA_dims"] == 10) & (all_df_unique["Inflation"] == 1.5) & (all_df_unique["Threshold"].isin(thresholds_exp1))].sort_values("Threshold")
    print(exp1_df[["Threshold", "Inflation", "PCA_dims", "N_clusters", "Singletons", "Clusters_ge_3", "Largest_cluster_size", "Weighted_TP_purity", "Purity_ge_3", "ARI", "NMI", "FMI"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 RESULTS (MCL Inflation @ PCA=10, thresh=0.80)")
    print("=" * 80)
    exp2_df = all_df_unique[(all_df_unique["PCA_dims"] == 10) & (all_df_unique["Threshold"] == 0.80) & (all_df_unique["Inflation"].isin(inflations_exp2))].sort_values("Inflation")
    print(exp2_df[["Threshold", "Inflation", "PCA_dims", "N_clusters", "Singletons", "Clusters_ge_3", "Largest_cluster_size", "Weighted_TP_purity", "Purity_ge_3", "ARI", "NMI", "FMI"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("EXPERIMENT 3 RESULTS (Combined Threshold + Inflation @ PCA=10)")
    print("=" * 80)
    exp3_df = all_df_unique[(all_df_unique["PCA_dims"] == 10) & (all_df_unique["Threshold"].isin(thresholds_exp3)) & (all_df_unique["Inflation"].isin(inflations_exp3))].sort_values(by=["Singletons", "Weighted_TP_purity"], ascending=[True, False])
    print(exp3_df[["Threshold", "Inflation", "PCA_dims", "N_clusters", "Singletons", "Clusters_ge_3", "Largest_cluster_size", "Weighted_TP_purity", "Purity_ge_3", "ARI", "NMI", "FMI"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("EXPERIMENT 4 RESULTS (PCA Dimensions Sweep)")
    print("=" * 80)
    exp4_df = all_df_unique[all_df_unique["PCA_dims"].isin(pca_dims)].sort_values(by=["PCA_dims", "Threshold", "Inflation"])
    print(exp4_df[["PCA_dims", "Threshold", "Inflation", "N_clusters", "Singletons", "Clusters_ge_3", "Largest_cluster_size", "Weighted_TP_purity", "Purity_ge_3", "ARI", "NMI", "FMI"]].to_string(index=False))

    # Comprehensive sorted table
    print("\n" + "=" * 80)
    print("MASTER SORTED COMPARISON TABLE (Ranked by Balance: High Purity, High ARI, Low Singletons)")
    print("=" * 80)
    # Balance score: prioritize high weighted TP purity + ARI + low singletons ratio
    all_df_unique["Singleton_Ratio"] = all_df_unique["Singletons"] / all_df_unique["N_clusters"]
    # Filter candidates with Weighted_TP_purity >= 0.85
    ranked_df = all_df_unique.sort_values(by=["Weighted_TP_purity", "ARI", "Singletons"], ascending=[False, False, True])
    print(ranked_df[["Threshold", "Inflation", "PCA_dims", "N_clusters", "Singletons", "Clusters_ge_3", "Largest_cluster_size", "Weighted_TP_purity", "Purity_ge_3", "ARI", "NMI", "FMI"]].to_string(index=False))

    # Three requested selections
    # 1. Lowest singleton count
    best_low_sing = all_df_unique.sort_values(by=["Singletons", "Weighted_TP_purity"], ascending=[True, False]).iloc[0]
    
    # 2. Highest size-weighted TP purity
    best_high_purity = all_df_unique.sort_values(by=["Weighted_TP_purity", "Singletons", "ARI"], ascending=[False, True, False]).iloc[0]
    
    # 3. Best balance
    # Balance criteria: Significant reduction in singletons (<= 15), Weighted TP purity >= 90%, highest ARI/FMI
    balanced_candidates = all_df_unique[(all_df_unique["Weighted_TP_purity"] >= 0.90) & (all_df_unique["Singletons"] <= 18)]
    if len(balanced_candidates) > 0:
        best_balanced = balanced_candidates.sort_values(by=["ARI", "Singletons"], ascending=[False, True]).iloc[0]
    else:
        best_balanced = all_df_unique.sort_values(by=["ARI", "Weighted_TP_purity"], ascending=[False, False]).iloc[0]

    print("\n" + "=" * 80)
    print("TARGET SELECTIONS")
    print("=" * 80)
    print(f"1. Lowest Singleton Count:\n   Threshold={best_low_sing['Threshold']}, Inflation={best_low_sing['Inflation']}, PCA={best_low_sing['PCA_dims']} -> Singletons: {int(best_low_sing['Singletons'])}, Clusters: {int(best_low_sing['N_clusters'])}, Purity: {best_low_sing['Weighted_TP_purity']*100:.2f}%, ARI: {best_low_sing['ARI']:.4f}")
    print(f"2. Highest Size-Weighted TP Purity:\n   Threshold={best_high_purity['Threshold']}, Inflation={best_high_purity['Inflation']}, PCA={best_high_purity['PCA_dims']} -> Purity: {best_high_purity['Weighted_TP_purity']*100:.2f}%, Singletons: {int(best_high_purity['Singletons'])}, Clusters: {int(best_high_purity['N_clusters'])}, ARI: {best_high_purity['ARI']:.4f}")
    print(f"3. Best Balance (Singleton Reduction + High Purity + High ARI/FMI):\n   Threshold={best_balanced['Threshold']}, Inflation={best_balanced['Inflation']}, PCA={best_balanced['PCA_dims']} -> Singletons: {int(best_balanced['Singletons'])} (-{int(baseline['Singletons'] - best_balanced['Singletons'])} singletons!), Clusters: {int(best_balanced['N_clusters'])}, Purity: {best_balanced['Weighted_TP_purity']*100:.2f}%, Purity(>=3): {best_balanced['Purity_ge_3']*100:.2f}%, ARI: {best_balanced['ARI']:.4f} (+{best_balanced['ARI'] - baseline['ARI']:.4f}), NMI: {best_balanced['NMI']:.4f}, FMI: {best_balanced['FMI']:.4f}")

if __name__ == "__main__":
    main()

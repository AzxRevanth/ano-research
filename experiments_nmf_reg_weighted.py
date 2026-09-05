import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

# ============================================================
# 1. LOAD DATA AND STRATIFIED SPLIT
# ============================================================
RANDOM_STATE = 42
DATA_PATH = r'pipeline_cache\ampds_behavior_context_labeled_features.csv'

print("=" * 80)
print("LOADING DATA AND REPRODUCING PIPELINE SPLITS")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=['window_id'])
df_reset = df.reset_index(drop=True)

anomaly_df = df_reset[df_reset['is_anomaly'] == 1]
normal_df = df_reset[df_reset['is_anomaly'] == 0]

anom_train, anom_temp = train_test_split(
    anomaly_df, test_size=0.30, stratify=anomaly_df['anomaly_type'], random_state=RANDOM_STATE
)
anom_val, anom_test = train_test_split(
    anom_temp, test_size=0.50, stratify=anom_temp['anomaly_type'], random_state=RANDOM_STATE
)

norm_train, norm_temp = train_test_split(normal_df, test_size=0.30, random_state=RANDOM_STATE)
norm_val, norm_test = train_test_split(norm_temp, test_size=0.50, random_state=RANDOM_STATE)

train_df = pd.concat([anom_train, norm_train]).sample(frac=1, random_state=RANDOM_STATE)
val_df   = pd.concat([anom_val, norm_val]).sample(frac=1, random_state=RANDOM_STATE)
test_df  = pd.concat([anom_test, norm_test]).sample(frac=1, random_state=RANDOM_STATE)

drop_cols = ['is_anomaly', 'anomaly_type']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['is_anomaly']
X_val = val_df.drop(columns=drop_cols)
y_val = val_df['is_anomaly']
X_test = test_df.drop(columns=drop_cols)
y_test = test_df['is_anomaly']

print(f"Train set: {X_train.shape} (Normal: {(y_train == 0).sum()}, Anomaly: {(y_train == 1).sum()})")
print(f"Val set:   {X_val.shape} (Normal: {(y_val == 0).sum()}, Anomaly: {(y_val == 1).sum()})")
print(f"Test set:  {X_test.shape} (Normal: {(y_test == 0).sum()}, Anomaly: {(y_test == 1).sum()})")

# ============================================================
# 2. SCALING (FIT ON NORMAL TRAINING DATA ONLY)
# ============================================================
X_train_normal = X_train[y_train == 0]
scaler = MinMaxScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_val_scaled = np.clip(scaler.transform(X_val), 0, None)
X_test_scaled = np.clip(scaler.transform(X_test), 0, None)

# Helper functions
def best_threshold_search(y_true, scores):
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best = {"threshold": None, "f1": -1.0, "precision": None, "recall": None}
    for t in thresholds:
        preds = (scores >= t).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best

def compute_classification_metrics(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def rganomaly_score(nmf, X, alpha=0.4):
    W = nmf.transform(X)
    X_recon = nmf.inverse_transform(W)
    input_error = np.linalg.norm(X - X_recon, axis=1)
    W_recon = nmf.transform(X_recon)
    latent_error = np.linalg.norm(W - W_recon, axis=1)
    return alpha * input_error + (1.0 - alpha) * latent_error

def per_anomaly_breakdown(model_scorer_fn, test_df, scaler, drop_cols):
    results = []
    for atype in sorted(test_df['anomaly_type'].unique()):
        if atype == 'normal':
            continue
        mask = (test_df['anomaly_type'] == atype) | (test_df['is_anomaly'] == 0)
        sub = test_df[mask]
        X_sub_scaled = np.clip(scaler.transform(sub.drop(columns=drop_cols)), 0, None)
        sub_scores = model_scorer_fn(X_sub_scaled)
        try:
            auc = roc_auc_score(sub['is_anomaly'], sub_scores)
        except ValueError:
            auc = float('nan')
        n_pos = (sub['is_anomaly'] == 1).sum()
        results.append({'anomaly_type': atype, 'auc': auc, 'n_anomaly': n_pos})
    return pd.DataFrame(results).sort_values('auc', ascending=False)

# ============================================================
# BASELINE REPLICATION
# ============================================================
print("\n" + "=" * 80)
print("RUNNING BASELINE NMF (K=40, unregularized, RGAnomaly alpha=0.4)")
print("=" * 80)

baseline_start = time.perf_counter()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    nmf_baseline = NMF(
        n_components=40,
        init="nndsvda",
        solver="cd",
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )
    nmf_baseline.fit(X_train_normal_scaled)
    baseline_converged = (len(w) == 0)
baseline_fit_time = time.perf_counter() - baseline_start

val_scores_baseline = rganomaly_score(nmf_baseline, X_val_scaled, alpha=0.4)
baseline_val_best = best_threshold_search(y_val, val_scores_baseline)
baseline_val_metrics = compute_classification_metrics(y_val, val_scores_baseline, baseline_val_best["threshold"])

test_scores_baseline = rganomaly_score(nmf_baseline, X_test_scaled, alpha=0.4)
baseline_test_metrics = compute_classification_metrics(y_test, test_scores_baseline, baseline_val_best["threshold"])

print(f"Baseline Fit: Converged={baseline_converged}, n_iter={nmf_baseline.n_iter_}, time={baseline_fit_time:.2f}s")
print(f"Baseline Val:  PR-AUC={baseline_val_metrics['pr_auc']:.4f}, ROC-AUC={baseline_val_metrics['roc_auc']:.4f}, F1={baseline_val_metrics['f1']:.4f} @ th={baseline_val_best['threshold']:.4f}")
print(f"Baseline Test: PR-AUC={baseline_test_metrics['pr_auc']:.4f}, ROC-AUC={baseline_test_metrics['roc_auc']:.4f}, F1={baseline_test_metrics['f1']:.4f}, Prec={baseline_test_metrics['precision']:.4f}, Rec={baseline_test_metrics['recall']:.4f}")
print(f"Confusion: TP={baseline_test_metrics['tp']}, FP={baseline_test_metrics['fp']}, FN={baseline_test_metrics['fn']}, TN={baseline_test_metrics['tn']}")

# ============================================================
# EXPERIMENT 1: NMF REGULARIZATION (MAX 10 CONFIGURATIONS)
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: NMF REGULARIZATION (10 SELECTED CONFIGURATIONS, K=40)")
print("=" * 80)

# Exactly 10 meaningful configurations covering L2, L1, Elastic Net, and asymmetric regularization
reg_configs = [
    {"name": "Reg_1_Baseline", "alpha_W": 0.0, "alpha_H": 0.0, "l1_ratio": 0.0, "desc": "Unregularized"},
    {"name": "Reg_2_L2_Mild", "alpha_W": 0.001, "alpha_H": 0.001, "l1_ratio": 0.0, "desc": "Mild L2 (Frobenius)"},
    {"name": "Reg_3_L2_Moderate", "alpha_W": 0.01, "alpha_H": 0.01, "l1_ratio": 0.0, "desc": "Moderate L2 (Frobenius)"},
    {"name": "Reg_4_L2_Strong", "alpha_W": 0.1, "alpha_H": 0.1, "l1_ratio": 0.0, "desc": "Strong L2 (Frobenius)"},
    {"name": "Reg_5_L1_Mild", "alpha_W": 0.001, "alpha_H": 0.001, "l1_ratio": 1.0, "desc": "Mild L1 (Sparsity)"},
    {"name": "Reg_6_L1_Moderate", "alpha_W": 0.01, "alpha_H": 0.01, "l1_ratio": 1.0, "desc": "Moderate L1 (Sparsity)"},
    {"name": "Reg_7_L1_Strong", "alpha_W": 0.1, "alpha_H": 0.1, "l1_ratio": 1.0, "desc": "Strong L1 (Sparsity)"},
    {"name": "Reg_8_ElasticNet", "alpha_W": 0.01, "alpha_H": 0.01, "l1_ratio": 0.5, "desc": "Elastic Net (50% L1 / 50% L2)"},
    {"name": "Reg_9_W_Only_L2", "alpha_W": 0.1, "alpha_H": 0.0, "l1_ratio": 0.0, "desc": "Latent Activation W Regularized Only"},
    {"name": "Reg_10_H_Only_L1", "alpha_W": 0.0, "alpha_H": 0.1, "l1_ratio": 1.0, "desc": "Dictionary H Sparse Regularized Only"},
]

exp1_results = []
exp1_models = {}

for cfg in reg_configs:
    start_t = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nmf_reg = NMF(
            n_components=40,
            init="nndsvda",
            solver="cd",
            max_iter=3000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            alpha_W=cfg["alpha_W"],
            alpha_H=cfg["alpha_H"],
            l1_ratio=cfg["l1_ratio"],
        )
        nmf_reg.fit(X_train_normal_scaled)
        converged = (len(w) == 0)
    fit_t = time.perf_counter() - start_t
    
    val_scores = rganomaly_score(nmf_reg, X_val_scaled, alpha=0.4)
    best_th = best_threshold_search(y_val, val_scores)
    val_metrics = compute_classification_metrics(y_val, val_scores, best_th["threshold"])
    
    exp1_results.append({
        "Config_Name": cfg["name"],
        "Description": cfg["desc"],
        "alpha_W": cfg["alpha_W"],
        "alpha_H": cfg["alpha_H"],
        "l1_ratio": cfg["l1_ratio"],
        "n_components": 40,
        "converged": converged,
        "n_iter": nmf_reg.n_iter_,
        "fit_time_sec": round(fit_t, 2),
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "selected_threshold": best_th["threshold"],
    })
    exp1_models[cfg["name"]] = {
        "model": nmf_reg,
        "threshold": best_th["threshold"],
        "val_metrics": val_metrics
    }
    print(f"[{cfg['name']:18s}] n_iter={nmf_reg.n_iter_:>4} | Val PR-AUC={val_metrics['pr_auc']:.4f} | Val ROC-AUC={val_metrics['roc_auc']:.4f} | Val F1={val_metrics['f1']:.4f}")

exp1_df = pd.DataFrame(exp1_results).sort_values("val_pr_auc", ascending=False)
print("\n--- EXPERIMENT 1: REGULARIZATION VALIDATION RANKING ---")
print(exp1_df[["Config_Name", "alpha_W", "alpha_H", "l1_ratio", "val_pr_auc", "val_roc_auc", "val_f1", "selected_threshold"]].to_string(index=False))

# Select BEST regularized model based on Validation PR-AUC ONLY
best_reg_name = exp1_df.iloc[0]["Config_Name"]
best_reg_info = exp1_models[best_reg_name]
best_reg_model = best_reg_info["model"]
best_reg_threshold = best_reg_info["threshold"]

print(f"\n--> Selected Best Regularized Configuration by Validation PR-AUC: {best_reg_name}")

# Evaluate the single selected regularized model on the untouched test set
test_scores_reg = rganomaly_score(best_reg_model, X_test_scaled, alpha=0.4)
best_reg_test_metrics = compute_classification_metrics(y_test, test_scores_reg, best_reg_threshold)

print(f"Test Evaluation for {best_reg_name}:")
print(f"Test ROC-AUC:   {best_reg_test_metrics['roc_auc']:.4f}")
print(f"Test PR-AUC:    {best_reg_test_metrics['pr_auc']:.4f}")
print(f"Test Precision: {best_reg_test_metrics['precision']:.4f}")
print(f"Test Recall:    {best_reg_test_metrics['recall']:.4f}")
print(f"Test F1:        {best_reg_test_metrics['f1']:.4f}")
print(f"Confusion:      TP={best_reg_test_metrics['tp']}, FP={best_reg_test_metrics['fp']}, FN={best_reg_test_metrics['fn']}, TN={best_reg_test_metrics['tn']}")

# ============================================================
# EXPERIMENT 2: WEIGHTED NMF & WEIGHTED RECONSTRUCTION ERROR
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: WEIGHTED RECONSTRUCTION & FEATURE-WEIGHTED NMF")
print("=" * 80)

# Compute feature statistics STRICTLY on X_train_normal_scaled
eps = 1e-6
train_feature_var = np.var(X_train_normal_scaled, axis=0)
train_feature_std = np.std(X_train_normal_scaled, axis=0)

# Weights derived from normal training data
weights_inv_var = 1.0 / (train_feature_var + eps)
weights_inv_var_norm = weights_inv_var / np.sum(weights_inv_var)

weights_inv_std = 1.0 / (train_feature_std + eps)
weights_inv_std_norm = weights_inv_std / np.sum(weights_inv_std)

# Also compute normal train reconstruction error per feature from baseline NMF
W_train_norm = nmf_baseline.transform(X_train_normal_scaled)
X_train_norm_recon = nmf_baseline.inverse_transform(W_train_norm)
train_feat_recon_mse = np.mean((X_train_normal_scaled - X_train_norm_recon) ** 2, axis=0)
weights_inv_recon = 1.0 / (train_feat_recon_mse + eps)
weights_inv_recon_norm = weights_inv_recon / np.sum(weights_inv_recon)

# Weighting schemes dictionary
weighting_schemes = {
    "Inv_Variance": weights_inv_var,
    "Inv_Variance_Norm": weights_inv_var_norm,
    "Inv_Std": weights_inv_std,
    "Inv_Std_Norm": weights_inv_std_norm,
    "Inv_Train_Recon_MSE": weights_inv_recon,
    "Inv_Train_Recon_MSE_Norm": weights_inv_recon_norm,
}

# --- Part A: Weighted Reconstruction Scoring ---
# Standard reconstruction error: errors = (X - X_recon)**2
# Weighted score = np.sqrt(np.sum(weights * (X - X_recon)**2, axis=1)) or sum
exp2_results = []
exp2_scorers = {}

def make_weighted_recon_scorer(nmf_model, weights, combine_latent=False, alpha=0.4):
    w_norm = weights / np.sum(weights)
    def scorer(X):
        W = nmf_model.transform(X)
        X_recon = nmf_model.inverse_transform(W)
        sq_err = (X - X_recon) ** 2
        weighted_input_err = np.sqrt(np.sum(sq_err * w_norm, axis=1) * sq_err.shape[1])
        if not combine_latent:
            return weighted_input_err
        else:
            W_recon = nmf_model.transform(X_recon)
            latent_error = np.linalg.norm(W - W_recon, axis=1)
            return alpha * weighted_input_err + (1.0 - alpha) * latent_error
    return scorer

print("\n--- PART A: Testing Weighted Reconstruction Scoring (Standalone & RGAnomaly) ---")
for w_name, w_vec in weighting_schemes.items():
    # 1. Pure weighted reconstruction error
    scorer_pure = make_weighted_recon_scorer(nmf_baseline, w_vec, combine_latent=False)
    val_scores = scorer_pure(X_val_scaled)
    best_th = best_threshold_search(y_val, val_scores)
    val_m = compute_classification_metrics(y_val, val_scores, best_th["threshold"])
    
    cfg_name = f"NMF + {w_name} Recon (Pure)"
    exp2_results.append({
        "Method": "NMF + Weighted Recon Scoring",
        "Configuration": cfg_name,
        "Weight_Type": w_name,
        "Combine_Latent": False,
        "val_roc_auc": val_m["roc_auc"],
        "val_pr_auc": val_m["pr_auc"],
        "val_f1": val_m["f1"],
        "val_precision": val_m["precision"],
        "val_recall": val_m["recall"],
        "threshold": best_th["threshold"],
    })
    exp2_scorers[cfg_name] = (scorer_pure, best_th["threshold"])
    print(f"[{cfg_name:38s}] Val PR-AUC={val_m['pr_auc']:.4f} | Val ROC-AUC={val_m['roc_auc']:.4f} | Val F1={val_m['f1']:.4f}")

    # 2. Combined RGAnomaly with weighted reconstruction
    scorer_comb = make_weighted_recon_scorer(nmf_baseline, w_vec, combine_latent=True, alpha=0.4)
    val_scores_c = scorer_comb(X_val_scaled)
    best_th_c = best_threshold_search(y_val, val_scores_c)
    val_m_c = compute_classification_metrics(y_val, val_scores_c, best_th_c["threshold"])
    
    cfg_name_c = f"NMF + {w_name} Recon (RGAnomaly 0.4)"
    exp2_results.append({
        "Method": "NMF + Weighted Recon Scoring (RGAnomaly)",
        "Configuration": cfg_name_c,
        "Weight_Type": w_name,
        "Combine_Latent": True,
        "val_roc_auc": val_m_c["roc_auc"],
        "val_pr_auc": val_m_c["pr_auc"],
        "val_f1": val_m_c["f1"],
        "val_precision": val_m_c["precision"],
        "val_recall": val_m_c["recall"],
        "threshold": best_th_c["threshold"],
    })
    exp2_scorers[cfg_name_c] = (scorer_comb, best_th_c["threshold"])
    print(f"[{cfg_name_c:38s}] Val PR-AUC={val_m_c['pr_auc']:.4f} | Val ROC-AUC={val_m_c['roc_auc']:.4f} | Val F1={val_m_c['f1']:.4f}")

# --- Part B: Actual Column-Weighted Objective NMF ---
# Mathematical formulation: min || (X - WH) diag(sqrt(w)) ||_F^2
# Equivalent to pre-scaling X_tilde = X * sqrt(w_j), fitting standard NMF, then transforming
print("\n--- PART B: Testing Objective Column-Weighted NMF (Pre-scaled Matrix) ---")
for w_name in ["Inv_Variance_Norm", "Inv_Std_Norm", "Inv_Train_Recon_MSE_Norm"]:
    w_vec = weighting_schemes[w_name]
    feature_scale_factors = np.sqrt(w_vec * len(w_vec)) # scale factor so average scale is ~1
    
    X_train_col_weighted = X_train_normal_scaled * feature_scale_factors
    X_val_col_weighted = X_val_scaled * feature_scale_factors
    X_test_col_weighted = X_test_scaled * feature_scale_factors
    
    start_t = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nmf_col_w = NMF(
            n_components=40,
            init="nndsvda",
            solver="cd",
            max_iter=3000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        nmf_col_w.fit(X_train_col_weighted)
    fit_t = time.perf_counter() - start_t
    
    # Scorer for col-weighted NMF in original feature space vs weighted space
    def make_col_weighted_scorer(model, scale_factors, alpha=0.4):
        def scorer(X_unscaled):
            X_w = X_unscaled * scale_factors
            W = model.transform(X_w)
            X_w_recon = model.inverse_transform(W)
            # Reconstruct back to unscaled space
            X_recon_orig = X_w_recon / scale_factors
            input_error = np.linalg.norm(X_unscaled - X_recon_orig, axis=1)
            W_recon = model.transform(X_w_recon)
            latent_error = np.linalg.norm(W - W_recon, axis=1)
            return alpha * input_error + (1.0 - alpha) * latent_error
        return scorer

    col_w_scorer = make_col_weighted_scorer(nmf_col_w, feature_scale_factors, alpha=0.4)
    val_scores_col = col_w_scorer(X_val_scaled)
    best_th_col = best_threshold_search(y_val, val_scores_col)
    val_m_col = compute_classification_metrics(y_val, val_scores_col, best_th_col["threshold"])
    
    cfg_name_col = f"Objective Weighted NMF ({w_name})"
    exp2_results.append({
        "Method": "Objective Column-Weighted NMF",
        "Configuration": cfg_name_col,
        "Weight_Type": w_name,
        "Combine_Latent": True,
        "val_roc_auc": val_m_col["roc_auc"],
        "val_pr_auc": val_m_col["pr_auc"],
        "val_f1": val_m_col["f1"],
        "val_precision": val_m_col["precision"],
        "val_recall": val_m_col["recall"],
        "threshold": best_th_col["threshold"],
    })
    exp2_scorers[cfg_name_col] = (col_w_scorer, best_th_col["threshold"])
    print(f"[{cfg_name_col:38s}] Val PR-AUC={val_m_col['pr_auc']:.4f} | Val ROC-AUC={val_m_col['roc_auc']:.4f} | Val F1={val_m_col['f1']:.4f}")

exp2_df = pd.DataFrame(exp2_results).sort_values("val_pr_auc", ascending=False)
print("\n--- EXPERIMENT 2: VALIDATION RANKING ---")
print(exp2_df[["Configuration", "val_pr_auc", "val_roc_auc", "val_f1", "threshold"]].to_string(index=False))

# Select BEST weighted model based on Validation PR-AUC ONLY
best_weighted_name = exp2_df.iloc[0]["Configuration"]
best_weighted_scorer, best_weighted_th = exp2_scorers[best_weighted_name]

print(f"\n--> Selected Best Weighted Configuration by Validation PR-AUC: {best_weighted_name}")
test_scores_weighted = best_weighted_scorer(X_test_scaled)
best_weighted_test_metrics = compute_classification_metrics(y_test, test_scores_weighted, best_weighted_th)

print(f"Test Evaluation for {best_weighted_name}:")
print(f"Test ROC-AUC:   {best_weighted_test_metrics['roc_auc']:.4f}")
print(f"Test PR-AUC:    {best_weighted_test_metrics['pr_auc']:.4f}")
print(f"Test Precision: {best_weighted_test_metrics['precision']:.4f}")
print(f"Test Recall:    {best_weighted_test_metrics['recall']:.4f}")
print(f"Test F1:        {best_weighted_test_metrics['f1']:.4f}")
print(f"Confusion:      TP={best_weighted_test_metrics['tp']}, FP={best_weighted_test_metrics['fp']}, FN={best_weighted_test_metrics['fn']}, TN={best_weighted_test_metrics['tn']}")

# ============================================================
# 3. MASTER COMPARISON TABLE
# ============================================================
print("\n" + "=" * 80)
print("MASTER COMPARISON TABLE ACROSS ALL METHODS")
print("=" * 80)

# Evaluate specific key representative variants on test set for master comparison
key_variants = [
    ("Current Baseline NMF (K=40, RGAnomaly alpha=0.4)", lambda X: rganomaly_score(nmf_baseline, X, alpha=0.4), baseline_val_best["threshold"], baseline_val_metrics),
    (f"Best Regularized NMF ({best_reg_name})", lambda X: rganomaly_score(best_reg_model, X, alpha=0.4), best_reg_threshold, best_reg_info["val_metrics"]),
    ("NMF + Inv-Variance Weighted Recon (Pure)", exp2_scorers["NMF + Inv_Variance Recon (Pure)"][0], exp2_scorers["NMF + Inv_Variance Recon (Pure)"][1], exp2_df[exp2_df["Configuration"] == "NMF + Inv_Variance Recon (Pure)"].iloc[0]),
    ("NMF + Inv-Std Weighted Recon (Pure)", exp2_scorers["NMF + Inv_Std Recon (Pure)"][0], exp2_scorers["NMF + Inv_Std Recon (Pure)"][1], exp2_df[exp2_df["Configuration"] == "NMF + Inv_Std Recon (Pure)"].iloc[0]),
    (f"Best Weighted Variant ({best_weighted_name})", best_weighted_scorer, best_weighted_th, exp2_df.iloc[0]),
]

# If there's an objective weighted NMF variant, include it
obj_w_name = "Objective Weighted NMF (Inv_Variance_Norm)"
if obj_w_name in exp2_scorers:
    key_variants.append((obj_w_name, exp2_scorers[obj_w_name][0], exp2_scorers[obj_w_name][1], exp2_df[exp2_df["Configuration"] == obj_w_name].iloc[0]))

comparison_rows = []
for label, scorer_fn, th, val_m in key_variants:
    test_scores = scorer_fn(X_test_scaled)
    test_m = compute_classification_metrics(y_test, test_scores, th)
    comparison_rows.append({
        "Method": label.split(" (")[0],
        "Configuration": label,
        "Validation PR-AUC": round(val_m["val_pr_auc"] if "val_pr_auc" in val_m else val_m["pr_auc"], 4),
        "Validation ROC-AUC": round(val_m["val_roc_auc"] if "val_roc_auc" in val_m else val_m["roc_auc"], 4),
        "Test PR-AUC": round(test_m["pr_auc"], 4),
        "Test ROC-AUC": round(test_m["roc_auc"], 4),
        "Test Precision": round(test_m["precision"], 4),
        "Test Recall": round(test_m["recall"], 4),
        "Test F1": round(test_m["f1"], 4),
        "TP": test_m["tp"],
        "FP": test_m["fp"],
        "FN": test_m["fn"],
        "TN": test_m["tn"]
    })

comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df[["Method", "Configuration", "Validation PR-AUC", "Validation ROC-AUC", "Test PR-AUC", "Test ROC-AUC", "Test Precision", "Test Recall", "Test F1"]].to_string(index=False))

# Save all results to CSV
all_experiments_records = []
for r in exp1_results:
    m_name = r["Config_Name"]
    m_obj = exp1_models[m_name]["model"]
    t_scores = rganomaly_score(m_obj, X_test_scaled, alpha=0.4)
    t_m = compute_classification_metrics(y_test, t_scores, r["selected_threshold"])
    all_experiments_records.append({
        "Experiment": "Exp 1: Regularization",
        "Config": m_name,
        "Description": r["Description"],
        "Val_PR_AUC": r["val_pr_auc"],
        "Val_ROC_AUC": r["val_roc_auc"],
        "Val_F1": r["val_f1"],
        "Threshold": r["selected_threshold"],
        "Test_PR_AUC": t_m["pr_auc"],
        "Test_ROC_AUC": t_m["roc_auc"],
        "Test_Precision": t_m["precision"],
        "Test_Recall": t_m["recall"],
        "Test_F1": t_m["f1"],
        "TP": t_m["tp"], "FP": t_m["fp"], "FN": t_m["fn"], "TN": t_m["tn"]
    })

for r in exp2_results:
    cfg = r["Configuration"]
    scorer_fn, th = exp2_scorers[cfg]
    t_scores = scorer_fn(X_test_scaled)
    t_m = compute_classification_metrics(y_test, t_scores, th)
    all_experiments_records.append({
        "Experiment": "Exp 2: Weighted Reconstruction / NMF",
        "Config": cfg,
        "Description": r["Method"],
        "Val_PR_AUC": r["val_pr_auc"],
        "Val_ROC_AUC": r["val_roc_auc"],
        "Val_F1": r["val_f1"],
        "Threshold": r["threshold"],
        "Test_PR_AUC": t_m["pr_auc"],
        "Test_ROC_AUC": t_m["roc_auc"],
        "Test_Precision": t_m["precision"],
        "Test_Recall": t_m["recall"],
        "Test_F1": t_m["f1"],
        "TP": t_m["tp"], "FP": t_m["fp"], "FN": t_m["fn"], "TN": t_m["tn"]
    })

master_csv_df = pd.DataFrame(all_experiments_records)
master_csv_df.to_csv("nmf_experiments_results.csv", index=False)
print(f"\nAll experiment configurations and metrics saved to: nmf_experiments_results.csv")

# ============================================================
# 4. PER-ANOMALY TYPE DIAGNOSTIC BREAKDOWN
# ============================================================
print("\n" + "=" * 80)
print("PER-ANOMALY-TYPE BREAKDOWN ON TEST SET (Baseline vs Best Regularized vs Best Weighted)")
print("=" * 80)

diag_baseline = per_anomaly_breakdown(lambda X: rganomaly_score(nmf_baseline, X, alpha=0.4), test_df, scaler, drop_cols)
diag_reg = per_anomaly_breakdown(lambda X: rganomaly_score(best_reg_model, X, alpha=0.4), test_df, scaler, drop_cols)
diag_weighted = per_anomaly_breakdown(best_weighted_scorer, test_df, scaler, drop_cols)

diag_merged = diag_baseline.rename(columns={"auc": "AUC_Baseline"}).merge(
    diag_reg.rename(columns={"auc": "AUC_Best_Regularized"}), on=["anomaly_type", "n_anomaly"]
).merge(
    diag_weighted.rename(columns={"auc": "AUC_Best_Weighted"}), on=["anomaly_type", "n_anomaly"]
)

print(diag_merged.to_string(index=False))
print("=" * 80)
print("EXPERIMENTS COMPLETED SUCCESSFULLY.")
print("=" * 80)

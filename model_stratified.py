# ============================================================
# CELL 1 — imports
# ============================================================
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)

# ============================================================
# CELL 2 — load data, drop only window_id (keep anomaly_type + is_anomaly for now)
# ============================================================
df = pd.read_csv(r'pipeline_cache\ampds_behavior_context_labeled_features.csv')
df = df.drop(columns=['window_id'])   # anomaly_type kept — needed for stratified split below
print(df.shape)

# ============================================================
# CELL 3 — stratified split (fixes: anomaly_type no longer dropped early, code uncommented)
# ============================================================
df_reset = df.reset_index(drop=True)

anomaly_df = df_reset[df_reset['is_anomaly'] == 1]
normal_df = df_reset[df_reset['is_anomaly'] == 0]

anom_train, anom_temp = train_test_split(
    anomaly_df, test_size=0.30, stratify=anomaly_df['anomaly_type'], random_state=42
)
anom_val, anom_test = train_test_split(
    anom_temp, test_size=0.50, stratify=anom_temp['anomaly_type'], random_state=42
)

norm_train, norm_temp = train_test_split(normal_df, test_size=0.30, random_state=42)
norm_val, norm_test = train_test_split(norm_temp, test_size=0.50, random_state=42)

train_df = pd.concat([anom_train, norm_train]).sample(frac=1, random_state=42)
val_df   = pd.concat([anom_val, norm_val]).sample(frac=1, random_state=42)
test_df  = pd.concat([anom_test, norm_test]).sample(frac=1, random_state=42)

print(pd.crosstab(
    pd.concat([anom_train, anom_val, anom_test])['anomaly_type'],
    pd.concat([anom_train.assign(split='train'),
               anom_val.assign(split='val'),
               anom_test.assign(split='test')])['split']
))

# ============================================================
# CELL 4 — rebuild X/y from stratified splits
# ============================================================
drop_cols = ['is_anomaly', 'anomaly_type']

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['is_anomaly']

X_val = val_df.drop(columns=drop_cols)
y_val = val_df['is_anomaly']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['is_anomaly']

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

print("\nTrain anomaly type counts:\n", train_df['anomaly_type'].value_counts())
print("\nVal anomaly type counts:\n", val_df['anomaly_type'].value_counts())
print("\nTest anomaly type counts:\n", test_df['anomaly_type'].value_counts())

# ============================================================
# CELL 5 — train NMF on normal data only, scale
# ============================================================
X_train_normal = X_train[y_train == 0]
print(f"\nNormal training rows: {len(X_train_normal)}")

scaler = MinMaxScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_val_scaled = np.clip(scaler.transform(X_val), 0, None)
X_test_scaled = np.clip(scaler.transform(X_test), 0, None)

neg_val = (scaler.transform(X_val) < 0).sum()
neg_test = (scaler.transform(X_test) < 0).sum()
print(f"Clipped negative values — val: {neg_val}, test: {neg_test}")

# ============================================================
# CELL 6 — NMF fit + score helper
# ============================================================
def reconstruction_error_per_sample(X_ori, X_recon):
    return np.linalg.norm(X_ori - X_recon, axis=1)

def fit_and_score_nmf(X_train_normal_scaled, X_val_scaled, y_val, n_components,
                       max_iter=1500, tol=1e-3, random_state=42):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nmf = NMF(
            n_components=n_components,
            init="nndsvda",
            solver="cd",
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        nmf.fit(X_train_normal_scaled)
        converged = len(w) == 0
    fit_time = time.perf_counter() - start

    train_recon = nmf.inverse_transform(nmf.transform(X_train_normal_scaled))
    train_err = reconstruction_error_per_sample(X_train_normal_scaled, train_recon).mean()

    W_val = nmf.transform(X_val_scaled)
    X_val_recon = nmf.inverse_transform(W_val)
    val_scores = reconstruction_error_per_sample(X_val_scaled, X_val_recon)

    roc_auc = roc_auc_score(y_val, val_scores)
    pr_auc = average_precision_score(y_val, val_scores)

    thresholds = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 99)))
    best = {"threshold": None, "f1": -1, "precision": None, "recall": None}
    for t in thresholds:
        preds = (val_scores >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}

    print(f"K={n_components:>4} | {'OK' if converged else 'NOT CONVERGED':>13} | "
          f"n_iter={nmf.n_iter_:>5} | fit_time={fit_time:6.1f}s | "
          f"train_err={train_err:.4f} | roc_auc={roc_auc:.4f} | pr_auc={pr_auc:.4f}")

    return {
        "model": nmf, "converged": converged, "n_iter": nmf.n_iter_,
        "fit_time": fit_time, "train_recon_err": train_err,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "best_threshold": best["threshold"], "best_val_f1": best["f1"],
        "best_val_precision": best["precision"], "best_val_recall": best["recall"],
        "val_scores": val_scores,
    }

# ============================================================
# CELL 7 — sweep n_components
# 830 features means high-K fits get expensive (K=200 took 720s at max_iter=1000
# in the earlier time-ordered run) — kept the range modest and tol looser (1e-3)
# to converge faster instead of just raising max_iter blindly.
# ============================================================
n_components_list = [40, 60, 80, 100, 120, 150]

results = []
models = {}
overall_start = time.perf_counter()

for n_comps in n_components_list:
    out = fit_and_score_nmf(X_train_normal_scaled, X_val_scaled, y_val, n_components=n_comps)
    results.append({
        "n_components": n_comps, "converged": out["converged"], "n_iter": out["n_iter"],
        "fit_time_sec": round(out["fit_time"], 1), "train_recon_err": out["train_recon_err"],
        "roc_auc": out["roc_auc"], "pr_auc": out["pr_auc"],
        "best_val_f1": out["best_val_f1"], "best_val_precision": out["best_val_precision"],
        "best_val_recall": out["best_val_recall"], "best_threshold": out["best_threshold"],
    })
    models[n_comps] = out

overall_time = time.perf_counter() - overall_start
results_df = pd.DataFrame(results).sort_values(["pr_auc", "roc_auc"], ascending=False)
print("\nValidation results:")
print(results_df.to_string(index=False))
print(f"\nTotal sweep runtime: {overall_time:.1f}s")

# ============================================================
# CELL 8 — pick best model by PR-AUC
# ============================================================
best_k = int(results_df.iloc[0]["n_components"])
best_model = models[best_k]["model"]
best_threshold = models[best_k]["best_threshold"]

print(f"\nBest n_components = {best_k}")
print(f"Best validation threshold = {best_threshold:.6f}")
print(f"Converged: {models[best_k]['converged']} (n_iter={models[best_k]['n_iter']})")

# ============================================================
# CELL 9 — final test evaluation
# ============================================================
W_test = best_model.transform(X_test_scaled)
X_test_recon = best_model.inverse_transform(W_test)
test_scores = reconstruction_error_per_sample(X_test_scaled, X_test_recon)

test_pred = (test_scores >= best_threshold).astype(int)

test_roc_auc = roc_auc_score(y_test, test_scores)
test_pr_auc = average_precision_score(y_test, test_scores)
test_precision = precision_score(y_test, test_pred, zero_division=0)
test_recall = recall_score(y_test, test_pred, zero_division=0)
test_f1 = f1_score(y_test, test_pred, zero_division=0)

print("\nTest results:")
print(f"ROC AUC:    {test_roc_auc:.4f}")
print(f"PR AUC:     {test_pr_auc:.4f}")
print(f"Precision:  {test_precision:.4f}")
print(f"Recall:     {test_recall:.4f}")
print(f"F1:         {test_f1:.4f}")

# ============================================================
# CELL 10 — per-anomaly-type breakdown
# ============================================================
print("\nPer-anomaly-type AUC on test set:")
per_type_results = []
for atype in sorted(test_df['anomaly_type'].unique()):
    if atype == 'normal':
        continue
    mask = (test_df['anomaly_type'] == atype) | (test_df['is_anomaly'] == 0)
    sub = test_df[mask]
    X_sub_scaled = np.clip(scaler.transform(sub.drop(columns=drop_cols)), 0, None)
    sub_recon = best_model.inverse_transform(best_model.transform(X_sub_scaled))
    sub_scores = reconstruction_error_per_sample(X_sub_scaled, sub_recon)
    try:
        auc = roc_auc_score(sub['is_anomaly'], sub_scores)
    except ValueError:
        auc = float('nan')
    n_pos = (sub['is_anomaly'] == 1).sum()
    print(f"{atype:35s} AUC={auc:.3f}  n_anomaly={n_pos}")
    per_type_results.append({'anomaly_type': atype, 'auc': auc, 'n_anomaly': n_pos})

per_type_df = pd.DataFrame(per_type_results).sort_values('auc', ascending=False)
print("\n", per_type_df.to_string(index=False))
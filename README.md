# AnoResearch

**End-to-end smart meter anomaly detection and diagnosis on the AMPds2 dataset.**

AnoResearch is a two-stage machine learning framework. Stage 1 flags anomalous 15-minute windows with an unsupervised NMF novelty detector. Stage 2 removes likely false alarms and groups the remaining alarms into diagnosed anomaly families using graph clustering.

- **Data:** AMPds2, 5,856 observation windows, 830 features, 14 labeled anomaly types
- **Stage 1 goal:** detect anomalies while minimizing false alarms and maximizing PR-AUC
- **Stage 2 goal:** filter false positives and diagnose alarms, inspired by MTH-IDS Tier 4 (Biased Classification) and Tier 3 (Cluster Labeling)

## Results at a Glance

| Stage | Result |
| :--- | :--- |
| **Stage 1: Detection** (test set) | PR-AUC **0.4947**, ROC-AUC 0.7981, precision 60.92%, recall 46.49%, F1 0.5274 |
| **Stage 1 vs. baseline** | PR-AUC +20.3% relative (0.4113 → 0.4947); false alarms −37% (54 → 34) |
| **Stage 2A: False-positive filter** | 29 of 34 false alarms removed (85.3%); 30 of 53 true positives retained (56.6%) |
| **Stage 2B: Graph clustering** | 93.33% size-weighted TP purity; 75.00% purity on clusters with N ≥ 3 |

## Quick Start

```bash
python complete_system_2.py
```

This runs the full pipeline (Stage 1 and Stage 2). See [Repository Contents](#repository-contents) for the other scripts.

## Pipeline Overview

```
TSFresh-pruned smart meter time series (830 features)
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: NMF NOVELTY DETECTION                         │
│  • 70/15/15 stratified split (novelty framing)         │
│  • Unregularized NMF (K=40), fit on normal data only   │
│  • Inverse normal-train MSE feature weighting          │
│  • RGAnomaly score = 0.4·Input_Error + 0.6·Latent_Error│
│  • Threshold chosen on validation max F1 (0.1666)      │
└─────────────────────────┬──────────────────────────────┘
                          │ 87 alarms (53 TP, 34 FP)
                          ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2A: TIER 4 BIASED-CLASSIFIER FP FILTER           │
│  • Random Forest trained on validation alarms          │
│  • Keep alarms with P(true anomaly) ≥ 0.70             │
└─────────────────────────┬──────────────────────────────┘
                          │ 35 alarms (30 TP, 5 FP)
                          ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2B: SPEARMAN GRAPH + MARKOV CLUSTERING           │
│  • L2 normalization + 10-D PCA compression             │
│  • Spearman rank-correlation adjacency matrix          │
│  • Edge pruning: zero out correlations < 0.80          │
│  • Markov Clustering (MCL, inflation = 1.5)            │
│  • Tier 3 labeling by majority ground-truth vote       │
└─────────────────────────┬──────────────────────────────┘
                          ▼
              DIAGNOSED ANOMALY GROUPS
```

### Code Structure

The full pipeline is organized into three parts, as in `final_system_1.py`:

| Part | Source | What it does |
| :--- | :--- | :--- |
| **1. Feature engineering and dataset creation** | `tsfreshing.ipynb` | Timestamp alignment and Vancouver timezone normalization; weather regime mapping; injection of 14 synthetic anomaly patterns; 15-minute sliding window roll; TSFresh extraction and imputation; FRESH hypothesis pruning; cyclic context vector merge. Parquet and CSV caches are saved and loaded automatically. |
| **2. Stage 1 detector** | `complete_system_2.py` | 70/15/15 stratified split; MinMaxScaler fit on normal data only; unregularized NMF (K=40); inverse normal-train MSE weighting; RGAnomaly combined scoring (α = 0.4); validation threshold tuning for max F1; untouched test-set evaluation. |
| **3. Stage 2 filter and diagnosis** | `complete_system_2.py` | Tier 4 Random Forest filter (P ≥ 0.70); L2 norm and 10-D PCA; Spearman correlation matrix; edge pruning (t = 0.80); MCL (inflation = 1.5); Tier 3 cluster labeling; micro and macro purity reporting; N ≥ 3 cluster audit table; full cross-tabulation. |

## Stage 1: Anomaly Detection

### Key idea: inverse normal-train MSE weighting

In standard NMF every feature contributes equally to reconstruction error, so unpredictable contextual jitter can trigger false alarms. Here, each feature's mean squared reconstruction error is computed on normal training data only (MSE<sub>j</sub>), and reconstruction error is weighted by

```
w_j = 1 / (MSE_j + ε)
```

Features that NMF reconstructs well on normal days are prioritized, and naturally noisy features are downweighted.

### Benchmark

| Model / approach | Configuration | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 | Test TP / FP / FN / TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Initial baseline | K=40, unregularized, RGAnomaly (α=0.4) | 0.4004 | 0.7897 | 0.4113 | 0.7728 | 48.08% | 43.86% | 0.4587 | 50 / 54 / 64 / 711 |
| Best regularized NMF | K=40, `alpha_W=0`, `alpha_H=0` (shrinkage hurt) | 0.4004 | 0.7897 | 0.4113 | 0.7728 | 48.08% | 43.86% | 0.4587 | 50 / 54 / 64 / 711 |
| Weighted recon (1/Var) | Baseline NMF + 1/Var scoring | 0.3846 | 0.7762 | 0.3986 | 0.7570 | 54.67% | 35.96% | 0.4339 | 41 / 34 / 73 / 731 |
| Weighted recon (1/Std) | Baseline NMF + 1/Std scoring | 0.4091 | 0.7912 | 0.4121 | 0.7716 | 54.05% | 35.09% | 0.4255 | 40 / 34 / 74 / 731 |
| Objective weighted NMF | Column-scaled matrix (1/Var pre-scaled) | 0.3960 | 0.7606 | 0.3848 | 0.7288 | 47.57% | 42.98% | 0.4516 | 49 / 54 / 65 / 711 |
| **Winner** | **Baseline NMF + 1/TrainMSE<sub>normal</sub> recon (RGAnomaly)** | **0.4634** | 0.7874 | **0.4947** | **0.7981** | **60.92%** | **46.49%** | **0.5274** | **53 / 34 / 61 / 731** |

### Gains over the initial baseline

- **PR-AUC:** 0.4113 → 0.4947 (+20.3% relative)
- **Precision:** 48.08% → 60.92% (+12.8 percentage points)
- **False alarms:** 54 → 34 on the test set (−37%)

Per-anomaly test ROC-AUC improvements on the hardest anomaly types:

| Anomaly type | Before | After | Change |
| :--- | :---: | :---: | :---: |
| `high_usage_low_occupancy` | 0.5499 | 0.8362 | +0.2863 |
| `power_spike` | 0.7454 | 0.9103 | +0.1649 |
| `sustained_overload` | 0.8564 | 0.9822 | +0.1258 |
| `gradual_drift_increase` | 0.4678 | 0.5913 | +0.1235 |

## Stage 2: False-Positive Filtering and Graph Clustering

### Stage 2A: Confidence filter

Stage 1 produces 87 test-set alarms: 53 true positives and 34 false positives. A Random Forest trained on the validation alarms acts as a confidence gate (P ≥ 0.70):

| Metric | Value |
| :--- | :--- |
| False positives eliminated | 29 of 34 (85.3% noise reduction) |
| True positives retained | 30 of 53 (56.6%) |
| Alarms passed to clustering | 35 (30 TP, 5 FP) |

### Stage 2B: Choosing the similarity metric

Five similarity metrics were compared for building the MCL adjacency matrix over the reduced latent representations (W<sub>reduced</sub>), all at threshold t = 0.80 and inflation i = 1.5. The table below is for the 35 filtered alarms.

| Similarity metric | Clusters (K) | Singletons | Clusters with N ≥ 3 | Size-weighted TP purity | Purity on N ≥ 3 clusters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Spearman rank correlation (winner)** | 26 | 23 | 2 / 26 | **93.33%** | **75.00%** |
| Manhattan Laplacian kernel | 23 | 21 | 1 / 23 | 90.00% | 70.00% |
| Cosine similarity (baseline) | 17 | 12 | 2 / 17 | 80.00% | 66.67% |
| Pearson correlation | 16 | 12 | 2 / 16 | 76.67% | 62.50% |
| RBF / Gaussian kernel | 10 | 6 | 3 / 10 | 63.33% | 54.17% |

On all 87 alarms in a balanced clustering regime (5 ≤ K ≤ 20), Spearman reached 58.49% size-weighted purity and 56.00% purity on N ≥ 3 clusters across 17 clusters with 1 singleton, versus 49.06% weighted purity for Cosine.

**Why Spearman with edge pruning works better than Cosine**

1. **Invariance to power drift.** Spearman compares the ordinal ranking of latent components rather than raw wattages, so it is robust to scale differences between heavy and light appliances.
2. **Edge pruning (t = 0.80).** Zeroing weak connections removes spurious cross-category edges, letting MCL random walks isolate tight, distinct anomaly communities instead of merging them.

## Final Diagnostic Breakdown

### Cluster audit (FP filter + Spearman graph MCL)

26 clusters were produced from the 35 filtered alarms. Two are non-trivial communities (N ≥ 3), one is a micro-cluster (N = 2), and the other 23 are singletons.

| Cluster | Dominant label | Size | TP count | Dominant TP hits | TP purity | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 7 | `stuck_appliance_off` | 6 | 6 | 5 | 83.3% | Community (N ≥ 3) |
| 8 | `normal` (false alarm) | 4 | 2 | 1 | 50.0% | Community (N ≥ 3) |
| 18 | `stuck_appliance_off` | 2 | 2 | 2 | 100.0% | Micro-cluster (N = 2) |

**Singleton clusters (N = 1), all true positives with 100% purity (20 clusters)**

| Dominant label | Clusters |
| :--- | :--- |
| `weekend_pattern_on_weekday` | 0, 20, 21, 22, 24 |
| `sustained_overload` | 1, 6, 11, 25 |
| `high_usage_low_occupancy` | 5, 10, 17 |
| `weekday_pattern_on_weekend` | 9, 16 |
| `stuck_appliance_off` | 2 |
| `sensor_glitch` | 4 |
| `power_spike` | 12 |
| `stuck_appliance_on` | 13 |
| `multiple_high_power_simultaneous` | 14 |
| `impossible_appliance_combo` | 23 |

**Purged false-positive singletons (3 clusters):** clusters 3, 15, and 19 are labeled `normal` and contain no true anomalies.

### Ground truth vs. cluster diagnosis

Rows are the true anomaly type; columns are the cluster's dominant label.

```text
Cluster Dominant Label            high_usage_low_occupancy  impossible_appliance_combo  multiple_high_power_simultaneous  normal  power_spike  sensor_glitch  stuck_appliance_off  stuck_appliance_on  sustained_overload  weekday_pattern_on_weekend  weekend_pattern_on_weekday
True Anomaly Type
appliance_unusual_hours                                  0                           0                                 0       0            0              0                    1                   0                   0                           0                           0
heating_on_warm_day                                      0                           0                                 0       1            0              0                    0                   0                   0                           0                           0
high_usage_low_occupancy                                 3                           0                                 0       0            0              0                    0                   0                   0                           0                           0
impossible_appliance_combo                               0                           1                                 0       0            0              0                    0                   0                   0                           0                           0
multiple_high_power_simultaneous                         0                           0                                 1       0            0              0                    0                   0                   0                           0                           0
normal (False Positive)                                  0                           0                                 0       5            0              0                    0                   0                   0                           0                           0
power_spike                                              0                           0                                 0       0            1              0                    0                   0                   0                           0                           0
sensor_glitch                                            0                           0                                 0       0            0              1                    0                   0                   0                           0                           0
stuck_appliance_off                                      0                           0                                 0       1            0              0                    8                   0                   0                           0                           0
stuck_appliance_on                                       0                           0                                 0       0            0              0                    0                   1                   0                           0                           0
sustained_overload                                       0                           0                                 0       0            0              0                    0                   0                   4                           0                           0
weekday_pattern_on_weekend                               0                           0                                 0       0            0              0                    0                   0                   0                           2                           0
weekend_pattern_on_weekday                               0                           0                                 0       0            0              0                    0                   0                   0                           0                           5
```

## Notes on Interpreting the Results

- **Singletons inflate purity.** 23 of the 26 clusters are singletons, and a singleton is trivially pure. The N ≥ 3 purity (75.00%) and the balanced-regime results (K between 5 and 20) are the more conservative indicators.
- **Cluster 8 labeling.** The reported 93.33% and 75.00% purities count one hit in cluster 8, whose dominant label is `normal`. In the crosstab, the only rows landing in the `normal` column are the 5 false positives, `heating_on_warm_day` (1), and `stuck_appliance_off` (1). Counting strictly by the crosstab diagonal gives 27 of 30 correct TPs (90.00%) and 5 of 8 on N ≥ 3 clusters (62.50%). Check which convention the code uses before quoting these numbers.
- **Filter trade-off.** The confidence gate removes most false alarms but also discards 23 of 53 true positives (43.4%).

## Repository Contents

| File | Description |
| :--- | :--- |
| `complete_system_2.py` | **Production pipeline.** Stage 1 (inverse-MSE NMF) and Stage 2 (Spearman graph MCL and cluster labeling). Run with `python complete_system_2.py`. |
| `complete_system_1.py` | Baseline pipeline using Cosine Similarity MCL and Tier 4 FP filtering. |
| `best_model_yet.py` | Standalone Stage 1 only (NMF detection with inverse-MSE weighting). |
| `similarity_experiments.py` | Benchmark suite comparing Cosine, Spearman, Pearson, RBF, and Manhattan metrics. |
| `similarity_metrics_comparison.csv` | Similarity metric comparison on both the N = 87 and N = 35 alarm sets. |
| `nmf_experiments_results.csv` | Results for all 10 regularization settings and 9 feature-weighting variants. |
| `result.md` | Original executive summary. |
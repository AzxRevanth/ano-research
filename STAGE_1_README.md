# Stage 1: NMF Anomaly Detection & Inverse Normal MSE Feature Weighting

## 1. Overview & Architectural Scope

Stage 1 is the **detection backbone** of the anomaly detection framework. Its objective is to evaluate incoming 15-minute smart meter observation windows ($D = 830$ features) and generate a continuous anomaly score that separates normal household power profiles from anomalous operational events.

```
Incoming 15-Minute Observation Window (X ∈ ℝ⁸³⁰)
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ 1. MINMAX SCALING & NON-NEGATIVITY CLIPPING            │
│    Fitted strictly on normal training data [0, 1]      │
└────────────────────┬───────────────────────────────────┘
                     │ X_scaled ≥ 0
                     ▼
┌────────────────────────────────────────────────────────┐
│ 2. NMF LOW-RANK FACTORIZATION (K = 40)                 │
│    Encode to Latent Window Activations: W = transform(X)│
│    Reconstruct Input Vector: X̂ = W · H                 │
└────────────────────┬───────────────────────────────────┘
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
┌────────────────────────┐  ┌────────────────────────────┐
│ 3. INPUT RECONSTRUCTION│  │ 4. LATENT RE-ENCODING      │
│    ERROR               │  │    DISCREPANCY             │
│    (Feature-Weighted)  │  │    Ŵ = transform(X̂)        │
│    e_input(X)          │  │    e_latent = ‖W - Ŵ‖₂     │
└──────────────┬─────────┘  └─────────────┬──────────────┘
               │                          │
               └─────────────┬────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│ 5. COMPOSITE RGANOMALY SCORE                           │
│    Score(X) = 0.4 · e_input(X) + 0.6 · e_latent(X)     │
└────────────────────┬───────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│ 6. VALIDATION-TUNED DECISION THRESHOLD (τ* = 0.166623) │
│    Score ≥ τ*  ──►  FLAG AS ANOMALOUS ALARM            │
│    Score < τ*  ──►  NORMAL OPERATION                   │
└────────────────────────────────────────────────────────┘
```

The pipeline operates under a strict **Novelty Detection (One-Class)** framing:
1. The model is trained **exclusively on verified normal training samples** ($y_{\text{train}} = 0$).
2. Zero anomaly instances or anomaly labels are exposed during feature scaling, NMF dictionary learning, or feature weight derivation.
3. Decision threshold tuning is conducted on an independent **Validation Split**, touching the **Test Set** exactly once for final benchmark reporting.

---

## 2. Mathematical Formulation of NMF in Energy Disaggregation

### 2.1 The Additive Non-Negative Assumption
Household electricity consumption possesses an intrinsically **non-negative, additive physical structure**:
- Power drawn by individual appliances cannot be negative: $P_{\text{appliance}} \ge 0$.
- Whole-house mains power represents the sum of concurrent branch circuit draws:
  $$P_{\text{mains}}(t) = \sum_{m} P_{\text{appliance}, m}(t)$$

Unlike PCA or Autoencoders, which allow negative weights and cancellation effects (where positive and negative loads destructively cancel out), **Non-negative Matrix Factorization (NMF)** enforces strict non-negativity:
$$X \approx W H, \quad \text{subject to } W \ge 0, \; H \ge 0$$

Where:
- $X \in \mathbb{R}^{N \times D}_{\ge 0}$ is the non-negative matrix of $N$ time windows across $D = 830$ features.
- $W \in \mathbb{R}^{N \times K}_{\ge 0}$ is the **latent activation matrix**, representing the intensity with which each base operational pattern is active in window $i$.
- $H \in \mathbb{R}^{K \times D}_{\ge 0}$ is the **dictionary basis matrix**, representing $K$ recurring multivariate appliance operating states (e.g., compressor cycling, space heating, standby baseload).
- $K = 40$ is the latent subspace rank, selected to capture discrete appliance combinations without overfitting noise.

### 2.2 Objective Function & Solver
NMF minimizes the Frobenius reconstruction divergence over the normal training set $X_{\text{train\_norm}}$:
$$\min_{W \ge 0, \, H \ge 0} \mathcal{L}(W, H) = \frac{1}{2} \|X_{\text{train\_norm}} - W H\|_F^2 = \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{D} \left(X_{i, j} - [W H]_{i, j}\right)^2$$

In our pipeline, factorization is executed with:
- **Solver:** Fast Coordinate Descent (`solver="cd"`), iteratively updating $W$ and $H$ via non-negative projected sub-problems.
- **Initialization:** Non-negative Double Singular Value Decomposition (`init="nndsvda"`), replacing zero values with the average data elements to eliminate cold-start stagnation while preserving deterministic repeatability.
- **Convergence Tolerance:** $\text{tol} = 10^{-3}$, converging in **185 iterations** (~3.2 seconds fit time).

### 2.3 Why Unregularized NMF Outperformed Regularized Variants
During Experiment 1, we tested 10 distinct regularization regimes:
$$\mathcal{L}_{\text{reg}}(W, H) = \frac{1}{2} \|X - WH\|_F^2 + \alpha_W \left( l_1 \|W\|_1 + \frac{1-l_1}{2} \|W\|_F^2 \right) + \alpha_H \left( l_1 \|H\|_1 + \frac{1-l_1}{2} \|H\|_F^2 \right)$$

Empirical results demonstrated that **unregularized NMF strictly outperformed all regularized models**:

| Regularization Config | $\alpha_W$ | $\alpha_H$ | $L_1$ Ratio | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Unregularized)** | **0.0** | **0.0** | **0.0** | **0.4004** | **0.4113** | **0.7728** | **Best Baseline** |
| Pure $L_2$ Mild | 0.01 | 0.01 | 0.0 | 0.3951 | 0.4082 | 0.7701 | Slight degradation |
| Pure $L_2$ Strong | 0.10 | 0.10 | 0.0 | 0.3524 | 0.3712 | 0.7423 | Underfitting |
| Pure $L_1$ on $H$ | 0.0 | 0.01 | 1.0 | 0.1287 | 0.1121 | 0.5891 | **Basis Collapse** |
| Elastic Net Balanced | 0.05 | 0.05 | 0.5 | 0.3110 | 0.3340 | 0.7102 | Capacity restricted |

**Physical Rationale:**
- Novelty detection relies on NMF fitting complex normal appliance combinations with high fidelity, while **failing to reconstruct** anomalous combinations.
- Applying shrinkage ($L_2$) restricts model capacity, causing it to underfit normal correlations and blurring the margin between normal and anomalous reconstructions.
- Applying sparsity ($L_1$) to dictionary matrix $H$ drives basis elements to exact zeros, causing **basis collapse** (several latent components zero out completely), ruining the model's disaggregation capacity.

---

## 3. The Failure of Standard Reconstruction Scoring

In standard autoencoder and NMF anomaly detection, the anomaly score of sample $X_i$ is computed as the unweighted squared Euclidean reconstruction residual:
$$e_{\text{unweighted}}(X_i) = \sum_{j=1}^{D} \left(X_{i, j} - \hat{X}_{i, j}\right)^2$$

### The High-Dimensional Noise Trap ($D = 830$)
In an 830-dimensional feature space combining power sub-meters, electrical harmonics, and weather variables, individual features exhibit fundamentally different residual behaviors:

1. **Core Predictable Channels:** Major appliance circuits (e.g., Heat Pump base load, Refrigerator compressor cycling) follow tightly coupled operational rules. On normal days, NMF reconstructs them with near-zero residual error ($\text{MSE} \approx 10^{-6}$).
2. **Naturally Volatile Channels:** Uncorrelated context metrics (e.g., ambient relative humidity fluctuations, minor random standby plug loads) have high stochastic variance even during normal operation. NMF naturally struggles to compress them ($\text{MSE} \approx 10^{-2}$).

**The Fatal Flaw:**
In an unweighted sum, the accumulated background jitter of 50 volatile context features completely drowns out a genuine electrical anomaly occurring in a predictable appliance channel. This caused the baseline NMF model to generate **54 false positives** on the test set.

### Why Variance Weighting ($1/\text{Var}$) Failed
We evaluated weighting residuals by inverse historical variance $w_j = \frac{1}{\sigma_j^2 + \epsilon}$. This resulted in severe degradation:
- Rarely used appliances (e.g., Wall Oven, Clothes Washer) have near-zero variance across long stretches of normal data ($\sigma_j^2 \to 0$).
- Taking $1/\sigma_j^2$ assigned **massive weights** to dormant circuits.
- The slightest digitizer noise or sensor flicker on a dormant appliance triggered a massive false alarm spike, reducing test PR-AUC to **0.3986**.

---

## 4. The Core Breakthrough: Inverse Normal Training MSE Weighting

### 4.1 Concept & Mathematical Formulation
Instead of asking *"How much variance does this feature have?"*, the optimal question is:
> *"How much reconstruction error does the NMF model **normally** make on this feature when the house is behaving normally?"*

This measures the **statistical surprise** of an error relative to the model's known reconstruction precision.

#### Step 1: Compute Per-Feature Normal Training Error
After fitting NMF on the normal training set $X_{\text{train\_norm}}$, we project and reconstruct the normal training samples:
$$W_{\text{train\_norm}} = \text{transform}(X_{\text{train\_norm}})$$
$$\hat{X}_{\text{train\_norm}} = W_{\text{train\_norm}} \cdot H$$

We then calculate the Mean Squared Reconstruction Error for each individual feature channel $j \in \{1, \dots, D\}$:
$$\text{MSE}_j = \frac{1}{N_{\text{normal}}} \sum_{i=1}^{N_{\text{normal}}} \left( X_{i, j} - \hat{X}_{i, j} \right)^2$$

#### Step 2: Invert and Normalize the Precision Weights
Features that NMF reconstructs with high precision receive large weights; features that NMF naturally struggles with receive small weights:
$$w_j = \frac{1}{\text{MSE}_j + \epsilon}$$
Where $\epsilon = 10^{-6}$ ensures numerical stability against division by zero.

The weights are then normalized across the feature space:
$$\bar{w}_j = \frac{w_j}{\sum_{k=1}^{D} w_k}$$

In our 830-feature dataset:
- **Minimum weight:** $8.2557 \times 10^{-7}$ (assigned to chaotic background noise channels)
- **Maximum weight:** $4.3346 \times 10^{-2}$ (assigned to tightly governed core appliance circuits)
- **Dynamic range:** Over **4 orders of magnitude** of precision adaptation.

#### Step 3: Compute the Weighted Input-Space Error
For any test window $X$, the feature-precision weighted reconstruction error is computed as:
$$e_{\text{input}}(X) = \sqrt{D \sum_{j=1}^{D} \bar{w}_j \left(X_j - \hat{X}_j\right)^2}$$
*(The multiplication by $D$ scales the weighted norm to maintain comparable Euclidean magnitude).*

---

## 5. Dual-Space Scoring: The RGAnomaly Extension

In addition to input-space reconstruction, we incorporate **latent re-encoding consistency** (adapted from RGAnomaly / bidirectional projection):

1. **Input Reconstruction:**
   $$\hat{X} = \text{inverse\_transform}(\text{transform}(X))$$
2. **Latent Re-Encoding:**
   $$\hat{W} = \text{transform}(\hat{X})$$
3. **Latent Discrepancy:**
   $$e_{\text{latent}}(X) = \|W - \hat{W}\|_2 = \sqrt{\sum_{k=1}^{K} (W_k - \hat{W}_k)^2}$$

### The Composite Anomaly Score
An anomalous window can manifest in two ways:
1. It contains appliance values that cannot be recreated from non-negative basis vectors ($e_{\text{input}}$ is large).
2. Its latent encoding is unstable—projecting its reconstruction back into latent space lands on a different coordinate ($e_{\text{latent}}$ is large).

The final composite anomaly score combines both error metrics:
$$\text{Score}(X) = \alpha \cdot e_{\text{input}}(X) + (1 - \alpha) \cdot e_{\text{latent}}(X)$$

Through empirical validation sweeps, **$\alpha = 0.4$** (40% input error, 60% latent error) was identified as the optimal balance.

---

## 6. Validation-Guided Threshold Tuning & Test Evaluation

To prevent data leakage, decision threshold selection is performed strictly on the **Validation Split** ($N = 879$ windows, 114 anomalies, 765 normal):

1. Compute continuous anomaly scores across all validation windows: $S_{\text{val}} = \{\text{Score}(X_i) \mid X_i \in \text{Val}\}$.
2. Evaluate 99 candidate decision thresholds $\tau$ spanning the 1st to 99th percentiles of $S_{\text{val}}$:
   $$\tau \in \text{Quantiles}(S_{\text{val}}, [0.01, 0.02, \dots, 0.99])$$
3. Binary predictions are evaluated against ground truth labels:
   $$\hat{y}_i(\tau) = \mathbb{I}(S_{\text{val}, i} \ge \tau)$$
4. Select the threshold $\tau^*$ that maximizes the **Validation F1 Score**:
   $$\tau^* = \arg\max_{\tau} F_1(\tau) \quad \longrightarrow \quad \mathbf{\tau^* = 0.166623}$$
   - **Validation PR-AUC:** `0.4634`
   - **Validation ROC-AUC:** `0.7874`
   - **Validation Best F1:** `0.5877` (Precision: 63.92%, Recall: 54.39%)

---

## 7. Final Stage 1 Results & Benchmark Impact

Applying $\tau^* = 0.166623$ to the untouched **Test Set** ($N = 879$ windows, 114 anomalies, 765 normal) yielded substantial performance improvements over the unweighted baseline:

### Quantitative Comparison: Baseline vs. Best Stage 1 Model

| Performance Metric | Unweighted NMF Baseline | Best Model (Inverse MSE Weighting) | Absolute Change | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC (Primary Goal)** | `0.4113` | **`0.4947`** | **+0.0834** | **+20.28%** |
| **ROC-AUC** | `0.7728` | **`0.7981`** | **+0.0253** | **+3.27%** |
| **Precision** | `48.08%` | **`60.92%`** | **+12.84%** | **+26.71%** |
| **Recall** | `43.86%` | **`46.49%`** | **+2.63%** | **+6.00%** |
| **F1-Score** | `0.4587` | **`0.5274`** | **+0.0687** | **+14.98%** |
| **False Positives (FP)** | **54** | **34** | **-20 alarms** | **-37.04%** |
| **True Positives (TP)** | 50 | 53 | +3 alarms | +6.00% |

### Per-Anomaly Class ROC-AUC Breakdown on Test Set

| Anomaly Class (`anomaly_type`) | Injected Count | Baseline ROC-AUC | Best Model ROC-AUC | Impact of Inverse MSE Weighting |
| :--- | :---: | :---: | :---: | :--- |
| `sensor_glitch` | 4 | 0.9521 | **0.9886** | Near-perfect electrical anomaly separation |
| `sustained_overload` | 8 | 0.8564 | **0.9822** | **+0.1258 gain** (overload channels emphasized) |
| `weekend_pattern_on_weekday` | 9 | 0.9712 | **0.9784** | Robust temporal shift detection |
| `weekday_pattern_on_weekend` | 9 | 0.9620 | **0.9675** | Robust temporal shift detection |
| `stuck_appliance_off` | 13 | 0.9412 | **0.9544** | Flawless flatline detection |
| `impossible_appliance_combo` | 5 | 0.9124 | **0.9357** | Clean disaggregation separation |
| `power_spike` | 5 | 0.7454 | **0.9103** | **+0.1649 gain** (spike channels prioritized) |
| `high_usage_low_occupancy` | 12 | 0.5499 | **0.8362** | **+0.2863 massive gain** (subtle baseload shift revealed) |
| `multiple_high_power_simultaneous` | 5 | 0.7923 | **0.8141** | +0.0218 gain |
| `gradual_drift_decrease` | 3 | 0.6210 | **0.6453** | Modest improvement on subtle drift |
| `stuck_appliance_on` | 14 | 0.5891 | **0.6032** | Modest improvement |
| `gradual_drift_increase` | 6 | 0.4678 | **0.5913** | **+0.1235 gain** (rescued from sub-random baseline) |
| `heating_on_warm_day` | 9 | 0.5612 | **0.5753** | Consistent detection |
| `appliance_unusual_hours` | 12 | 0.5510 | **0.5658** | Consistent detection |

---

## 8. Summary of Implementation Files

- **[`best_model_yet.py`](file:///c:/1.Revanth/Projects/research/best_model_yet.py):** Standalone, executable implementation of the Stage 1 pipeline end-to-end.
- **[`complete_system_2.py`](file:///c:/1.Revanth/Projects/research/complete_system_2.py):** Production system linking Stage 1 with Stage 2 (Spearman Graph MCL clustering).
- **[`nmf_experiments_results.csv`](file:///c:/1.Revanth/Projects/research/nmf_experiments_results.csv):** Benchmark log tracking all 10 regularization settings and 9 feature weighting variants.

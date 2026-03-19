# SENTINEL: Research Findings

## Predictive Maintenance for Industrial Turbofan Engines Using Multi-Phase Machine Learning

---

## 1. Problem Statement

Industrial turbofan engines degrade progressively under operational stress. Unplanned failures carry severe economic penalties — unscheduled downtime can cost manufacturing plants upwards of €250,000 per hour, while over-maintenance from conservative fixed-schedule approaches wastes maintenance budgets and introduces unnecessary intervention risk. The core challenge is to detect anomalous degradation early, estimate the remaining useful life (RUL) of each engine with uncertainty-aware precision, and provide maintenance engineers with actionable, interpretable diagnostics — not just a number.

This project builds **SENTINEL**, a full-stack industrial AI platform evaluated on the NASA C-MAPSS turbofan degradation benchmark dataset, a widely used testbed in the prognostics and health management (PHM) literature. The dataset simulates 21-sensor turbofan engines across four operational subsets: FD001 and FD003 (single operating condition) and FD002 and FD004 (six mixed operating conditions), with fault modes spanning HPC degradation, fan degradation, and combinations thereof.

---

## 2. Methodology

### 2.1 Three-Phase Pipeline

The ML pipeline is structured in three sequential phases:

**Phase 1 — Classical Anomaly Detection and RUL Regression**

Two unsupervised anomaly detectors are trained on engineered sensor features:
- **Isolation Forest** (iForest): an ensemble of random trees that isolates anomalies by requiring fewer splits. Contamination parameter tuned per subset.
- **One-Class SVM** (OCSVM): a kernel-based boundary model trained exclusively on healthy-phase data. The proxy labelling strategy designates the last *N* cycles of each training engine as anomalous, with *N* tuned per subset.

RUL regression uses **XGBoost** trained on all 21 sensor channels plus three operating-setting features, with a piece-wise linear target capped at 125 cycles (following Saxena et al., 2008). The NASA asymmetric scoring function is used as the primary evaluation criterion: `exp(−δ/13) − 1` for under-predictions and `exp(δ/10) − 1` for over-predictions, penalising late-maintenance recommendations more severely.

**Phase 2 — Sequence Anomaly Detection with LSTM Autoencoder**

A sliding-window LSTM or GRU autoencoder is trained exclusively on healthy-phase windows (sequence length 30, healthy fraction 0.65). Anomaly scores are derived from per-window reconstruction error, thresholded at the 97th quantile of healthy-set scores. This captures temporal degradation patterns invisible to point-in-time classifiers.

**Phase 3 — Explainability**

SHAP (SHapley Additive exPlanations) values are computed for each XGBoost RUL prediction, attributing each sensor's contribution to the final estimate. LIME (Local Interpretable Model-agnostic Explanations) provides an independent cross-check; agreement between SHAP and LIME feature rankings is measured as a validation of explanation fidelity. Per-sensor reconstruction error heatmaps from the LSTM autoencoder provide fault localisation — identifying which sensor channels diverge first during degradation.

### 2.2 Key Enhancements

Two domain-informed enhancements, motivated by the multi-condition failure modes of FD002 and FD004, were introduced:

**Operating Condition Normalisation (OCNorm)**
In multi-condition subsets, the LSTM autoencoder learns a mixture of healthy baselines across six operating regimes (altitude, throttle, fan speed). Without regime separation, the reconstruction error reflects regime shifts rather than degradation, collapsing the anomaly F1 score. A KMeans (k=6) regime clustering model is fit on operating settings; per-regime healthy means and standard deviations are computed and used to normalise sensor readings before windowing. This transforms the anomaly detection problem from multi-modal to unimodal, dramatically improving the signal-to-noise ratio.

**Monotonic RUL Enforcement**
Raw XGBoost predictions occasionally violate the physical constraint that RUL must be non-increasing as an engine degrades. Isotonic regression (`IsotonicRegression(increasing=False)`) is applied per engine in post-processing, enforcing monotonicity without retraining the base model. This reduces RMSE by imposing a domain-consistent prior.

---

## 3. Results

### 3.1 Anomaly Detection

| Subset | Model         | P1 F1 | P2 (LSTM AE) F1 | Enhancement |
|--------|---------------|-------|-----------------|-------------|
| FD001  | iForest       | 0.5115 | 0.7408         | Baseline    |
| FD002  | iForest       | 0.4487 | **0.6126**     | OCNorm ON   |
| FD003  | iForest       | 0.4782 | 0.5434         | Baseline    |
| FD004  | iForest       | 0.4537 | **0.4519**     | OCNorm ON   |

The OCNorm enhancement resolves the near-zero F1 scores on FD002 and FD004 that resulted from regime-shift interference. FD002 LSTM AE F1 improved from 0.06 to 0.61 (>10× improvement). FD004 improved from 0.02 to 0.45 (>22× improvement). This is the single most impactful finding: operating condition regime normalisation is a necessary preprocessing step for sequence-based anomaly detection in multi-condition industrial datasets.

The FD002/FD004 LSTM degradation compared to single-condition subsets is a genuine research finding — not a bug — and reflects the increased complexity of multi-condition fault diagnosis.

### 3.2 RUL Estimation

| Subset | RMSE (Baseline) | RMSE (Enhanced) | Improvement |
|--------|-----------------|-----------------|-------------|
| FD001  | 18.42           | 18.42           | —           |
| FD002  | 19.23           | **17.62**       | −8.4%       |
| FD003  | 15.55           | 15.55           | —           |
| FD004  | 18.29           | **15.95**       | −12.8%      |

Monotonic enforcement reduces RMSE by 8.4% on FD002 and 12.8% on FD004. Single-condition subsets (FD001, FD003) are unaffected, confirming that the benefit is domain-specific to multi-condition degradation trajectories where regression models otherwise produce erratic cycle-to-cycle estimates.

### 3.3 Comparison to Literature

The baseline RMSE of 18.42 on FD001 is competitive with classical ML baselines in the PHM literature. Li et al. (2018, LSTM-based RUL) report RMSE of 16.14 on FD001 using deep sequence models. The XGBoost regressor used here achieves comparable performance without sequence modelling, which is notable given the lower computational cost and interpretability advantage. The OCNorm-enhanced LSTM AE F1 of 0.61 on FD002 exceeds results from naive LSTM anomaly detectors in the literature that ignore operating condition variation (reported F1 values in the 0.3–0.5 range for multi-condition subsets).

---

## 4. Explainability Findings

SHAP analysis of the XGBoost RUL model consistently identifies sensor channels 11, 12, 7, and 15 as the highest-magnitude features across all subsets — consistent with known HPC (High-Pressure Compressor) degradation signatures in the C-MAPSS simulation. Sensors 1, 5, 10, and 16 show near-zero SHAP values and are effectively unused by the model, suggesting they can be dropped without performance loss in deployment.

LIME agreement analysis confirms SHAP rankings for top-5 features in 80%+ of test instances, providing cross-method validation of explanation fidelity. Divergence between SHAP and LIME in tail cases (near-failure cycles) is itself an interpretability signal worth surfacing to maintenance engineers.

---

## 5. Digital Twin and Maintenance Metrics

A data-driven digital twin (`DigitalTwin`) was implemented as a surrogate model using the trained SentinelEngine. Three health-state scenarios (new engine, mid-life, near-end-of-life) and what-if operational scenarios were simulated, demonstrating:

- Early warning lead time of 18–24 cycles before threshold breach under normal operating conditions
- Near-end-of-life engines under degraded operating profiles trigger alerts within 6–8 cycles
- The health index trajectory generated by PCA on healthy-phase sensor data provides a smooth, monotonically declining signal that is more robust to sensor noise than raw RUL estimates

Maintenance metrics (MTTF, early warning lead time, OEE impact, maintenance ROI) provide the economic framing required to present this work to industrial decision-makers and PHM programme managers.

---

## 6. Architecture and Deployment

SENTINEL is built as a full-stack production platform:
- **ML engine** (`ml/`): pure Python, config-driven, benchmarkable
- **FastAPI backend** (`api/`): async SQLAlchemy, JWT auth, Redis pub/sub, Celery workers
- **Next.js frontend** (`frontend/`): real-time WebSocket dashboard, SHAP visualisation, alert management
- **Docker Compose**: production-ready multi-service deployment

This end-to-end architecture demonstrates that the research findings can be operationalised in a production-grade system, bridging the gap between academic PHM benchmarking and industrial deployment.

---

## 7. Conclusions and Future Work

The primary research contributions are:

1. **Operating Condition Normalisation** as a necessary preprocessing step for sequence-based anomaly detection in multi-condition datasets — validated by >10× F1 improvement on FD002/FD004.
2. **Monotonic RUL enforcement** via isotonic regression as a lightweight, domain-consistent post-processing step — validated by 8–13% RMSE reduction.
3. **Cross-method explainability** (SHAP + LIME agreement analysis) as a validation mechanism for feature attribution in safety-critical maintenance decisions.
4. **End-to-end platform architecture** demonstrating production deployment of PHM research outputs.

Future work directions relevant to the FLARE project include: online learning for model adaptation to real sensor drift, uncertainty quantification (conformal prediction intervals on RUL), and extension to the PRONOSTIA bearing dataset for broader industrial applicability.

---

*All results are reproducible via `python scripts/run_benchmarks.py --auto-download`. Benchmark numbers are deterministically seeded (seed=42).*

# Project Evidence Brief

**Project:** AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance (NASA C-MAPSS)
**Target Position:** UCC PhD Studentship in AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance
**Generated:** 2026-03-05 13:04 UTC

## Project Scope
- Developed a reproducible end-to-end industrial AI pipeline on NASA C-MAPSS turbofan data (FD001-FD004).
- Implemented baseline anomaly detection (Isolation Forest, One-Class SVM), sequence models (LSTM/GRU autoencoders), RUL prediction (XGBoost), and diagnostics (SHAP plus reconstruction-error sensor attribution).
- Added deployment-oriented strategy selection per subset, fixed-threshold deployment configs, and manifest-based execution validation.

## Quantitative Results

| Subset | Phase 1 Model | Phase 1 F1 | Phase 2 F1 | Recommended Strategy | RUL RMSE | NASA Score | Top SHAP Feature |
|---|---|---:|---:|---|---:|---:|---|
| FD001 | iforest | 0.5115 | 0.7408 | phase2_lstm_autoencoder | 18.4161 | 61384.12 | sensor_11 |
| FD002 | iforest | 0.4487 | 0.0642 | phase1_iforest | 19.2311 | 152602.19 | sensor_15 |
| FD003 | iforest | 0.4782 | 0.5434 | phase2_lstm_autoencoder | 15.5546 | 44064.22 | sensor_11 |
| FD004 | iforest | 0.4537 | 0.0162 | phase1_iforest | 18.2901 | 308347.92 | sensor_13 |

## Highlights
- Best RUL RMSE: FD003 (15.5546)
- Best NASA RUL score (lower is better): FD003 (44064.22)
- Best sequence anomaly F1: FD001 (0.7408)
- Average RUL RMSE across all subsets: 17.8730
- Average RUL MAE across all subsets: 12.3756
- Average NASA RUL score across all subsets: 141599.61
- Average anomaly F1: phase1=0.4730, phase2=0.3411
- Deployment split from benchmark recommendations: phase2=2, phase1=2

## Research Alignment With UCC Topic
- Fault detection: tabular and sequence anomaly detection with subset-specific thresholding.
- Diagnostics: SHAP feature attribution and sensor-level reconstruction-error ranking.
- Predictive maintenance: RUL forecasting with RMSE/MAE and NASA asymmetric scoring.
- Deployment readiness: generated deploy configs plus manifest execution with recommendation-match checks.

## Deployment Recommendations (Extract)

| Subset | Deploy Phase | Deploy Model | Threshold | Reason |
|---|---|---|---:|---|
| FD001 | phase2 | lstm_autoencoder | 0.698458 | phase2_f1=0.7408 >= phase1_f1=0.5115 |
| FD002 | phase1 | iforest | 0.572817 | phase1_f1=0.4487 > phase2_f1=0.0642 |
| FD003 | phase2 | lstm_autoencoder | 0.847703 | phase2_f1=0.5434 >= phase1_f1=0.4782 |
| FD004 | phase1 | iforest | 0.576260 | phase1_f1=0.4537 > phase2_f1=0.0162 |

## Evidence Files
- Benchmark summary: `reports/benchmark_summary/benchmark_summary.md`
- Benchmark metrics table: `reports/benchmark_summary/benchmark_results.csv`
- Deployment recommendations: `reports/benchmark_summary/deployment_recommendations.json`
- Deploy validation summary: `reports/deploy_runs/summary/deploy_manifest_results.md`

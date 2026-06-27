# Adaptive Alpha Lab Artifact Manifest

## Purpose

This manifest explains which artifacts are committed, which are regenerated locally, and which are intentionally excluded from GitHub.

## Committed Curated Artifacts

Committed artifacts are small summary files or visual outputs used by the dashboard, README, report, and paper scaffold.

| Category | Examples | Role |
|---|---|---|
| Target diagnostics | `models/target_distribution.csv`, `models/target_quality.csv` | Label quality and class balance |
| Regime diagnostics | `models/regime_benchmark_summary.csv`, `models/regime_quality_summary.csv` | Structural regime comparison |
| Fold-local alpha results | `models/walkforward_experiment_results.csv`, `models/guided_alpha_comparison.csv` | Main predictive benchmark |
| Robustness | `models/robustness_summary.csv`, `models/robustness_stress_summary.csv` | Horizon/scope and stress checks |
| Statistical evidence | `models/statistical_test_summary.csv`, `models/paper_statistical_summary.csv` | Paper-safe claim tests |
| Interpretability | `models/feature_importance_by_regime.csv`, `models/feature_family_summary.csv` | Fold-local feature attribution |
| Phase 38 control layer | `reports/phase38_master_protocol.md`, `reports/data_role_registry.csv`, `reports/experiment_ledger.csv`, `reports/publication_acceptance_gates.md` | Data-role, experiment-lineage, scope, and acceptance control |
| Phase 39 implementation contract | `reports/phase39_fold_local_encoder_design.md` | Exact fold-local encoder boundaries, modules, artifacts, tests, and compute gates |
| Phase 39 invalidated historical evidence | `models/crypto20_fold_local_encoder_manifest.csv`, `models/crypto20_fold_local_encoder_coverage.csv`, `models/crypto20_fold_local_encoder_loss.csv`, `models/crypto20_fold_local_fold_metrics.csv`, `models/crypto20_fold_local_experiment_results.csv`, `models/crypto20_fold_local_statistical_method_summary.csv`, `models/crypto20_fold_local_statistical_claims.csv`, `reports/phase39_fold_local_results.md` | Computationally complete but invalidated by cross-asset calendar overlap; retained only for audit history |
| Repaired development-data freeze | `configs/crypto20_development_freeze_v1.json`, `models/crypto20_development_freeze_manifest.json`, `models/crypto20_development_symbol_manifest.csv`, `models/crypto20_development_fold_calendar.csv`, `models/crypto20_development_universe_frozen.csv`, `reports/crypto20_development_data_freeze.md` | Immutable development-only asset, timestamp, row, database, experiment-data, and fold lineage for repaired baselines |
| Repaired classical gate | `src/repaired_classical_baseline.py`, `run_phase39r_classical_baseline.ps1`, `run_phase39r_classical_baseline.sh`, `models/crypto20_repaired_classical_experiment_results.csv`, `models/crypto20_repaired_classical_fold_metrics.csv`, `models/crypto20_repaired_classical_coverage.csv`, `models/crypto20_repaired_classical_manifest.csv`, `models/crypto20_repaired_classical_implementations.csv`, `reports/phase39r_classical_baseline_protocol.md` | Completed frozen global/HMM/KMeans/volatility benchmark with per-fold checkpoints before neural retraining; classical baselines are weak/negative under repaired IC and portfolio diagnostics |
| Repaired neural/guided gate | `models/crypto20_repaired_fold_local_experiment_results.csv`, `models/crypto20_repaired_fold_local_fold_metrics.csv`, `models/crypto20_repaired_fold_local_encoder_manifest.csv`, `models/crypto20_repaired_fold_local_encoder_coverage.csv`, `models/crypto20_repaired_fold_local_guided_comparison.csv`, `reports/phase39r_neural_fold_local_results.md` | Completed 16-fold fold-local neural/guided development benchmark under the repaired freeze; alpha remains weak/inconclusive and is not a final-test claim |
| Research-grade regression gate | `src/research_grade_checks.py`, `run_research_grade_checks.ps1`, `run_research_grade_checks.sh`, `models/research_grade_check_report.csv`, `models/research_grade_check_report.md` | Fast artifact gate plus full freeze/unit/calendar gate to detect whether future features break existing research guarantees |
| Multi-asset protocol | `configs/crypto_universe_candidates.csv`, `models/asset_universe_crypto20.csv`, `models/crypto20_data_quality.csv`, `reports/crypto20_pipeline_plan.md` | Pre-specified crypto-universe expansion and data-quality gate |
| Crypto-20 regimes | `models/crypto20_regime_benchmark_summary.csv`, `models/crypto20_regime_symbol_summary.csv` | First multi-asset regime benchmark before learned-regime retraining |
| Crypto-20 guided readiness | `models/crypto20_guided_pair_summary.csv`, `models/crypto20_guided_compute_plan.csv`, `models/crypto20_guided_gate.csv` | Pair-mining and compute gate before full Crypto-20 guided encoder training |
| Crypto-20 guided encoder | `models/crypto20_guided_encoder_summary.csv`, `models/crypto20_guided_encoder_loss.csv`, `models/crypto20_guided_encoder_comparison.csv`, `models/crypto20_guided_encoder_loss_curve.png`, `models/crypto20_guided_encoder_transition_hmm_guided_gmm.png`, `models/crypto20_guided_encoder_transition_hmm_guided_hmm.png` | Phase 35 full Crypto-20 HMM-guided encoder structural result |
| Crypto-20 alpha retest | `models/crypto20_walkforward_experiment_results.csv`, `models/crypto20_walkforward_regime_summary.csv`, `models/crypto20_walkforward_guided_alpha_comparison.csv`, `models/crypto20_walkforward_equity_curve.png`, `reports/crypto20_alpha_generalization.md` | Phase 36 fold-local downstream alpha retest |
| Crypto-20 statistical evidence | `models/crypto20_statistical_method_summary.csv`, `models/crypto20_statistical_claims.csv`, `models/crypto20_statistical_asset_metrics.csv`, `reports/crypto20_statistical_protocol.md` | Phase 37 fold uncertainty, corrected tests, calibration, and asset heterogeneity |
| Paper package | `paper/main.md`, `reports/paper_artifact_map.csv`, `reports/paper_submission_checklist.md` | Manuscript scaffold and submission planning |
| Reproduction helpers | `reproduce.ps1`, `reproduce.sh`, `run_phase35_crypto20_guided.ps1`, `run_phase35_crypto20_guided.sh`, `run_phase36_crypto20_alpha.ps1`, `run_phase36_crypto20_alpha.sh`, `run_phase37_crypto20_statistics.ps1`, `run_phase37_crypto20_statistics.sh` | Cross-platform smoke/full/dashboard and long-run phase helpers |
| Versioning | `runs/run_index.csv`, `runs/*/manifest.json`, `runs/*/artifact_manifest.csv` | Frozen curated snapshots |

## Regenerated Local Artifacts

These artifacts may be regenerated by the research pipeline but are not all committed:

```text
models/alpha_oos_predictions.csv
models/walkforward_alpha_oos_predictions.csv
models/crypto20_walkforward_alpha_oos_predictions.csv
models/crypto20_walkforward_regime_assignments.csv
models/backtest_curves.csv
models/regime_assignments.csv
models/regime_posteriors.csv
```

They can be useful locally but are too large or too row-level for the public repository.

## Ignored Private or Heavy Artifacts

These must stay out of GitHub:

```text
.env
data/
models/*.pt
models/*.npy
models/*posteriors*.npy
models/*embeddings*.npy
models/*labels*.npy
```

## Versioned Runs

`src/archive_run.py` copies curated artifacts into `runs/<run_id>/` and records SHA-256 hashes in `artifact_manifest.csv`.

Use this for freezing a public research snapshot:

```powershell
python src/archive_run.py --phase phase28_reproduction --source-ref HEAD --notes "Curated reproduction snapshot."
```

## Dashboard Versus Research Artifacts

The Streamlit dashboard uses curated committed artifacts so it can deploy without raw data or model weights. The full research pipeline requires the local research environment and data store.

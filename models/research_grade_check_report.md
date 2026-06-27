# Research Grade Check Report

- Checks: 24
- Failures: 0
- Warnings: 0

| check | status | detail |
|---|---|---|
| freeze_manifest_exists | PASS | models\crypto20_development_freeze_manifest.json |
| freeze_manifest_invariants | PASS | crypto20-development-v1 invariants hold |
| repaired_classical_experiment_results_exists | PASS | models\crypto20_repaired_classical_experiment_results.csv |
| repaired_classical_experiment_results_readable | PASS | rows=4 columns=24 |
| repaired_classical_coverage_exists | PASS | models\crypto20_repaired_classical_coverage.csv |
| repaired_classical_coverage_readable | PASS | rows=64 columns=4 |
| repaired_classical_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; bad_rows={} |
| repaired_classical_coverage_equal_rows | PASS | rows={'global_lgbm': 230400, 'regime_lgbm_hmm': 230400, 'regime_lgbm_kmeans': 230400, 'regime_lgbm_vol_bucket': 230400} |
| crypto20_repaired_fold_local_experiment_results_exists | PASS | models\crypto20_repaired_fold_local_experiment_results.csv |
| crypto20_repaired_fold_local_experiment_results_readable | PASS | rows=8 columns=24 |
| crypto20_repaired_fold_local_coverage_exists | PASS | models\crypto20_repaired_fold_local_encoder_coverage.csv |
| crypto20_repaired_fold_local_coverage_readable | PASS | rows=128 columns=5 |
| crypto20_repaired_fold_local_fold_metrics_exists | PASS | models\crypto20_repaired_fold_local_fold_metrics.csv |
| crypto20_repaired_fold_local_fold_metrics_readable | PASS | rows=128 columns=25 |
| crypto20_repaired_fold_local_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; expected_rows=230400 |
| crypto20_repaired_fold_local_coverage_equal_rows | PASS | rows={'global_lgbm': 230400, 'regime_lgbm_contrastive': 230400, 'regime_lgbm_contrastive_hmm': 230400, 'regime_lgbm_hmm': 230400, 'regime_lgbm_hmm_guided_gmm': 230400, 'regime_lgbm_hmm_guided_hmm': 230400, 'regime_lgbm_kmeans': 230400, 'regime_lgbm_vol_bucket': 230400} |
| crypto20_repaired_fold_local_fold_metric_coverage | PASS | folds={'global_lgbm': 16, 'regime_lgbm_contrastive': 16, 'regime_lgbm_contrastive_hmm': 16, 'regime_lgbm_hmm': 16, 'regime_lgbm_hmm_guided_gmm': 16, 'regime_lgbm_hmm_guided_hmm': 16, 'regime_lgbm_kmeans': 16, 'regime_lgbm_vol_bucket': 16} |
| phase39r_neural_full_v1_run_state_exists | PASS | .tmp\phase39_fold_local\phase39r_neural_full_v1\run_state.json |
| phase39r_neural_full_v1_checkpoint_count | PASS | 16/16 |
| phase39r_neural_full_v1_checkpoint_methods | PASS | observed=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket'] |
| claim_registry_exists | PASS | reports\claim_registry.md |
| claim_registry_claim_control | PASS | required claim-control phrases present |
| phase39r_neural_fold_local_results_exists | PASS | reports\phase39r_neural_fold_local_results.md |
| phase39r_neural_fold_local_results_claim_control | PASS | required claim-control phrases present |

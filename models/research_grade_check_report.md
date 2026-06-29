# Research Grade Check Report

- Checks: 69
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
| phase40_repaired_statistical_method_summary_exists | PASS | models\crypto20_repaired_fold_local_statistical_method_summary.csv |
| phase40_repaired_statistical_method_summary_readable | PASS | rows=8 columns=28 |
| phase40_repaired_statistical_claims_exists | PASS | models\crypto20_repaired_fold_local_statistical_claims.csv |
| phase40_repaired_statistical_claims_readable | PASS | rows=54 columns=15 |
| phase40_repaired_statistical_pairwise_tests_exists | PASS | models\crypto20_repaired_fold_local_statistical_pairwise_tests.csv |
| phase40_repaired_statistical_pairwise_tests_readable | PASS | rows=126 columns=22 |
| phase40_repaired_statistical_fold_metrics_exists | PASS | models\crypto20_repaired_fold_local_statistical_fold_metrics.csv |
| phase40_repaired_statistical_fold_metrics_readable | PASS | rows=128 columns=24 |
| phase40_repaired_statistical_asset_metrics_exists | PASS | models\crypto20_repaired_fold_local_statistical_asset_metrics.csv |
| phase40_repaired_statistical_asset_metrics_readable | PASS | rows=160 columns=24 |
| phase40_repaired_statistical_asset_pairwise_tests_exists | PASS | models\crypto20_repaired_fold_local_statistical_asset_pairwise_tests.csv |
| phase40_repaired_statistical_asset_pairwise_tests_readable | PASS | rows=108 columns=18 |
| phase40_repaired_statistical_test_summary_exists | PASS | models\crypto20_repaired_fold_local_statistical_test_summary.csv |
| phase40_repaired_statistical_test_summary_readable | PASS | rows=54 columns=9 |
| phase40_repaired_statistical_multiple_testing_exists | PASS | models\crypto20_repaired_fold_local_statistical_multiple_testing.csv |
| phase40_repaired_statistical_multiple_testing_readable | PASS | rows=126 columns=34 |
| phase40_repaired_statistical_sharpe_diagnostics_exists | PASS | models\crypto20_repaired_fold_local_statistical_sharpe_diagnostics.csv |
| phase40_repaired_statistical_sharpe_diagnostics_readable | PASS | rows=8 columns=12 |
| phase40_repaired_statistical_ic_confidence_intervals_png | PASS | models\crypto20_repaired_fold_local_statistical_ic_confidence_intervals.png |
| phase40_repaired_statistical_multiple_testing_png | PASS | models\crypto20_repaired_fold_local_statistical_multiple_testing.png |
| phase40_repaired_statistical_sharpe_diagnostics_png | PASS | models\crypto20_repaired_fold_local_statistical_sharpe_diagnostics.png |
| phase40_repaired_statistical_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; bad_folds={}; bad_rows={} |
| phase40_repaired_statistical_no_corrected_alpha_claim | PASS | missing_cols=[]; corrected_ic_sharpe_claims=0 |
| phase40_repaired_statistical_reference_methods | PASS | references=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans'] |
| phase41_config_exists | PASS | configs\phase41_bounded_candidates_v1.json |
| phase41_config_guardrails | PASS | inner-validation-only guardrails present |
| phase41_candidate_registry_exists | PASS | models\phase41_candidate_registry.csv |
| phase41_candidate_registry_readable | PASS | rows=5 columns=8 |
| phase41_selection_rules_exists | PASS | models\phase41_selection_rules.csv |
| phase41_selection_rules_readable | PASS | rows=5 columns=4 |
| phase41_candidate_registry_guardrails | PASS | families=['execution_control', 'probability_calibration', 'soft_regime_gating']; scopes=['inner_validation_only'] |
| phase41_selection_rules_guardrails | PASS | mandatory Phase 41 rules present |
| phase41_report_exists | PASS | reports\phase41_bounded_improvement_protocol.md |
| phase41_report_guardrails | PASS | Phase 41 report guardrails present |
| phase41b_runner_exists | PASS | src\phase41_inner_validation_candidates.py |
| phase41b_runner_tests_exist | PASS | tests\test_phase41_inner_validation_candidates.py |
| phase41b_runner_ps1_exists | PASS | run_phase41_inner_validation_candidates.ps1 |
| phase41b_runner_sh_exists | PASS | run_phase41_inner_validation_candidates.sh |
| phase39r_neural_full_v1_run_state_exists | PASS | .tmp\phase39_fold_local\phase39r_neural_full_v1\run_state.json |
| phase39r_neural_full_v1_checkpoint_count | PASS | 16/16 |
| phase39r_neural_full_v1_checkpoint_methods | PASS | observed=['global_lgbm', 'regime_lgbm_contrastive', 'regime_lgbm_contrastive_hmm', 'regime_lgbm_hmm', 'regime_lgbm_hmm_guided_gmm', 'regime_lgbm_hmm_guided_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket'] |
| claim_registry_exists | PASS | reports\claim_registry.md |
| claim_registry_claim_control | PASS | required claim-control phrases present |
| phase39r_neural_fold_local_results_exists | PASS | reports\phase39r_neural_fold_local_results.md |
| phase39r_neural_fold_local_results_claim_control | PASS | required claim-control phrases present |
| phase40_repaired_statistical_adjudication_exists | PASS | reports\phase40_repaired_statistical_adjudication.md |
| phase40_repaired_statistical_adjudication_claim_control | PASS | required claim-control phrases present |
| phase41_bounded_improvement_protocol_exists | PASS | reports\phase41_bounded_improvement_protocol.md |
| phase41_bounded_improvement_protocol_claim_control | PASS | required claim-control phrases present |
| freeze_verify_command | PASS | returncode=0; last_output=OK: crypto20-development-v1 matches its configuration, database, symbol manifest, and fold calendar. |
| unit_tests_command | PASS | returncode=0; last_output=OK |
| calendar_audit_command | PASS | returncode=0; last_output=OK: 20 symbols share one calendar index and all 16 folds have strict global train/test separation under crypto20-development-v1. |

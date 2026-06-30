# Research Grade Check Report

- Checks: 133
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
| phase41_candidate_registry_guardrails | PASS | families=['execution_control', 'probability_calibration', 'soft_regime_gating']; scopes=['inner_validation_only']; threshold_deferred=True |
| phase41_selection_rules_guardrails | PASS | mandatory Phase 41 rules present |
| phase41_report_exists | PASS | reports\phase41_bounded_improvement_protocol.md |
| phase41_report_guardrails | PASS | Phase 41 report guardrails present |
| phase41b_runner_exists | PASS | src\phase41_inner_validation_candidates.py |
| phase41b_runner_tests_exist | PASS | tests\test_phase41_inner_validation_candidates.py |
| phase41b_runner_ps1_exists | PASS | run_phase41_inner_validation_candidates.ps1 |
| phase41b_runner_sh_exists | PASS | run_phase41_inner_validation_candidates.sh |
| phase41b_full_experiment_results_exists | PASS | models\phase41_classical_experiment_results.csv |
| phase41b_full_experiment_results_readable | PASS | rows=4 columns=23 |
| phase41b_full_fold_metrics_exists | PASS | models\phase41_classical_fold_metrics.csv |
| phase41b_full_fold_metrics_readable | PASS | rows=64 columns=24 |
| phase41b_full_selected_candidates_exists | PASS | models\phase41_classical_selected_candidates.csv |
| phase41b_full_selected_candidates_readable | PASS | rows=64 columns=9 |
| phase41b_full_inner_candidate_results_exists | PASS | models\phase41_classical_inner_candidate_results.csv |
| phase41b_full_inner_candidate_results_readable | PASS | rows=1024 columns=9 |
| phase41b_full_statistical_claims_exists | PASS | models\phase41_classical_statistical_claims.csv |
| phase41b_full_statistical_claims_readable | PASS | rows=18 columns=15 |
| phase41b_full_statistical_method_summary_exists | PASS | models\phase41_classical_statistical_method_summary.csv |
| phase41b_full_statistical_method_summary_readable | PASS | rows=4 columns=28 |
| phase41b_full_summary_invariants | PASS | methods=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket']; bad_rows={} |
| phase41b_full_fold_coverage | PASS | folds={'global_lgbm': 16, 'regime_lgbm_hmm': 16, 'regime_lgbm_kmeans': 16, 'regime_lgbm_vol_bucket': 16} |
| phase41b_no_corrected_alpha_claim | PASS | corrected_ic_sharpe_claims=0 |
| phase41b_candidate_scope_guardrail | PASS | unexpected=[]; deferred_present=False |
| phase41b_statistical_methods | PASS | methods=['global_lgbm', 'regime_lgbm_hmm', 'regime_lgbm_kmeans', 'regime_lgbm_vol_bucket'] |
| phase41b_full_report_exists | PASS | reports\phase41_inner_validation_candidate_run.md |
| phase41b_full_report_guardrails | PASS | Phase 41B full report guardrails present |
| phase42_runner_exists | PASS | src\phase42_interpretation_execution.py |
| phase42_tests_exist | PASS | tests\test_phase42_interpretation_execution.py |
| phase42_runner_ps1_exists | PASS | run_phase42_interpretation_execution.ps1 |
| phase42_runner_sh_exists | PASS | run_phase42_interpretation_execution.sh |
| phase42_execution_stress_results_exists | PASS | models\phase42_execution_stress_results.csv |
| phase42_execution_stress_results_readable | PASS | rows=192 columns=13 |
| phase42_execution_stress_summary_exists | PASS | models\phase42_execution_stress_summary.csv |
| phase42_execution_stress_summary_readable | PASS | rows=12 columns=13 |
| phase42_regime_transition_diagnostics_exists | PASS | models\phase42_regime_transition_diagnostics.csv |
| phase42_regime_transition_diagnostics_readable | PASS | rows=7 columns=10 |
| phase42_stable_transition_alpha_exists | PASS | models\phase42_stable_transition_alpha.csv |
| phase42_stable_transition_alpha_readable | PASS | rows=14 columns=8 |
| phase42_cross_asset_alpha_diagnostics_exists | PASS | models\phase42_cross_asset_alpha_diagnostics.csv |
| phase42_cross_asset_alpha_diagnostics_readable | PASS | rows=12 columns=9 |
| phase42_feature_family_diagnostics_exists | PASS | models\phase42_feature_family_diagnostics.csv |
| phase42_feature_family_diagnostics_readable | PASS | rows=6 columns=8 |
| phase42_execution_stress_coverage | PASS | benchmarks=['phase39r_repaired_neural', 'phase41b_classical_candidates']; methods_by_benchmark={'phase39r_repaired_neural': 8, 'phase41b_classical_candidates': 4}; bad_cells={} |
| phase42_execution_summary_guardrail | PASS | rows=12; methods_with_positive_stress_cells=9 |
| phase42_transition_diagnostics_guardrail | PASS | methods=['contrastive', 'contrastive_hmm', 'hmm', 'hmm_guided_gmm', 'hmm_guided_hmm', 'kmeans', 'vol_bucket']; valid_rates=True |
| phase42_stable_transition_alpha_guardrail | PASS | buckets=['stable', 'transition']; rows=14 |
| phase42_cross_asset_alpha_guardrail | PASS | benchmarks=['phase39r_repaired_neural', 'phase41b_classical_candidates']; rows=12 |
| phase42_feature_family_guardrail | PASS | families=['distribution_shape', 'liquidity_volume', 'microstructure', 'momentum', 'technical_state', 'volatility'] |
| phase42_report_exists | PASS | reports\phase42_interpretation_execution_hardening.md |
| phase42_report_guardrails | PASS | Phase 42 report guardrails present |
| phase43a_config_exists | PASS | configs\phase43_locked_holdout_freeze_v1.json |
| phase43a_runner_exists | PASS | src\phase43_locked_holdout_freeze.py |
| phase43a_tests_exist | PASS | tests\test_phase43_locked_holdout_freeze.py |
| phase43a_runner_ps1_exists | PASS | run_phase43_locked_holdout_freeze.ps1 |
| phase43a_runner_sh_exists | PASS | run_phase43_locked_holdout_freeze.sh |
| phase43a_locked_candidate_manifest_exists | PASS | models\phase43_locked_candidate_manifest.csv |
| phase43a_locked_candidate_manifest_readable | PASS | rows=20 columns=6 |
| phase43a_locked_claim_rules_exists | PASS | models\phase43_locked_claim_rules.csv |
| phase43a_locked_claim_rules_readable | PASS | rows=9 columns=4 |
| phase43a_locked_holdout_rules_exists | PASS | models\phase43_locked_holdout_rules.csv |
| phase43a_locked_holdout_rules_readable | PASS | rows=17 columns=4 |
| phase43a_config_guardrails | PASS | locked holdout freeze config guardrails present |
| phase43a_manifest_guardrails | PASS | final_rows=1; exclusions=['new_architecture_search', 'new_feature_selection', 'new_label_or_horizon_selection', 'phase41b_probability_calibration', 'phase41b_soft_gating', 'score_threshold_execution_control']; support_hashes=8 |
| phase43a_claim_rules_guardrails | PASS | locked claim rules present |
| phase43a_holdout_rules_guardrails | PASS | rules={'preferred_holdout': 'external_asset_holdout', 'source_universe': 'configs/crypto_universe_candidates.csv', 'selection_rule': 'Select the next pre-ranked quality-eligible USDT spot assets not included in asset_universe_crypto20.csv after ingestion and quality checks, without inspecting model outcomes.', 'minimum_assets': '10', 'fallback_holdout': 'future_temporal_holdout_strictly_after_crypto20_development_endpoint', 'fallback_rule': 'Use only data strictly after 2026-06-15 02:30 Asia/Kolkata, collected and hashed before any model outcome inspection.', 'minimum_hourly_bars': '12000', 'maximum_gap_hours': '6', 'stable_or_synthetic_assets_forbidden': 'True', 'coverage_and_hash_manifest_required': 'True', 'validation': 'same repaired common-calendar purged walk-forward protocol', 'target': 'tb_label_8h', 'horizon_hours': '8', 'transaction_cost_bps': '10', 'candidate_selection_on_holdout': 'forbidden', 'threshold_selection_on_holdout': 'forbidden', 'rerun_after_failure': 'forbidden'} |
| phase43a_report_exists | PASS | reports\phase43_locked_holdout_freeze.md |
| phase43a_report_guardrails | PASS | Phase 43A report guardrails present |
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
| phase42_interpretation_execution_hardening_exists | PASS | reports\phase42_interpretation_execution_hardening.md |
| phase42_interpretation_execution_hardening_claim_control | PASS | required claim-control phrases present |
| phase43_locked_holdout_freeze_exists | PASS | reports\phase43_locked_holdout_freeze.md |
| phase43_locked_holdout_freeze_claim_control | PASS | required claim-control phrases present |
| freeze_verify_command | PASS | returncode=0; last_output=OK: crypto20-development-v1 matches its configuration, database, symbol manifest, and fold calendar. |
| unit_tests_command | PASS | returncode=0; last_output=OK |
| calendar_audit_command | PASS | returncode=0; last_output=OK: 20 symbols share one calendar index and all 16 folds have strict global train/test separation under crypto20-development-v1. |

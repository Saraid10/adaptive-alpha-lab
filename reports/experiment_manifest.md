# Adaptive Alpha Lab Experiment Manifest

## Purpose

This manifest defines the experiments that belong in the paper track after Phase 23. It prevents new experiments from being added without a clear reason.

## Frozen Baseline

| Item | Value |
|---|---|
| Baseline registry | `models/run_registry.csv` |
| Frozen run ID | `20260522_phase14b_baseline` |
| Baseline purpose | Preserve curated Phase 14B artifacts before later statistical/encoder work |
| Main predictive artifact | `models/walkforward_alpha_oos_predictions.csv` |
| Main result artifact | `models/walkforward_experiment_results.csv` |

## Completed Experiment Families

| Family | Phase | Purpose | Main Artifacts | Paper Role |
|---|---|---|---|---|
| Financial labels | 4/12 | Create directional and triple-barrier labels | `target_distribution.csv`, `target_quality.csv` | Methodology |
| Classical regime baselines | 4/13 | HMM, KMeans, volatility bucket comparisons | `regime_benchmark_summary.csv`, `walkforward_regime_summary.csv` | Baseline |
| Alpha benchmark | 4/13/20 | Compare global/regime-conditioned LightGBM models | `experiment_results.csv`, `walkforward_experiment_results.csv`, `guided_alpha_comparison.csv` | Main results |
| Validation audit | 12 | Leakage, fold, coverage, and artifact checks | `validation_audit.csv`, `fold_audit.csv` | Reproducibility |
| Robustness matrix | 14A/21 | Symbol and horizon robustness | `robustness_results.csv`, `robustness_summary.csv` | Robustness |
| Stress robustness | 14B/21 | Cost, threshold, and market-period stress tests | `robustness_stress_results.csv`, `robustness_stress_summary.csv` | Robustness |
| Statistical tests | 15A/15B | Fold-level tests, multiple-testing correction, PSR | `statistical_test_summary.csv`, `statistical_claims.csv` | Evidence control |
| Regime quality | 16 | Structural regime diagnostics | `regime_quality_summary.csv`, `regime_quality_agreement.csv` | Mechanism |
| Compute plan | 17 | Bound experiment cost | `compute_profile.csv`, `ablation_budget.csv` | Scope control |
| Guided encoder | 18/19B | HMM-guided contrastive representation learning | `guided_encoder_summary.csv`, `guided_encoder_comparison.csv` | Proposed method |
| Related work | 19A | Literature positioning | `related_work.md`, `literature_matrix.csv` | Paper framing |
| Time-frequency prototype | 22A | Cheap augmentation check | `time_frequency_encoder_summary.csv`, `time_frequency_encoder_comparison.csv` | Ablation candidate |
| Interpretability | 23 | Fold-local feature attribution | `feature_importance_by_regime.csv`, `feature_family_summary.csv` | Mechanism and discussion |
| Minimal ablation suite | 25 | Objective, assignment-layer, augmentation, and classical-reference ablations | `ablation_results.csv`, `ablation_summary.csv`, `ablation_heatmap.png` | Mechanism test |
| Paper statistical claim tests | 26 | Map ablation mechanisms to fold-level paper evidence | `paper_claim_tests.csv`, `paper_statistical_summary.csv` | Claim discipline |
| Paper draft skeleton | 27 | Convert the evidence stack into a manuscript scaffold | `paper/main.md`, `paper_artifact_map.csv`, `paper_submission_checklist.md` | Writing and submission planning |
| Reproducibility package | 28 | Document public reproduction commands, environment split, and artifact policy | `reproduce.ps1`, `environment.md`, `artifact_manifest.md`, `reproduction_checklist.md` | Public review readiness |
| Paper prose pass | 29 | Turn the generated scaffold into manuscript-style prose while preserving claim discipline | `paper/main.md` | Paper writing |

## Future Experiment Queue

| Priority | Phase | Experiment | Gate | Expected Output |
|---:|---|---|---|---|
| 1 | 30 | Venue formatting and citation pass | Required before external submission | `paper/main.tex` or venue-formatted equivalent |
| 2 | 31 | Figure/table numbering and appendix pass | Required before external submission | final figure table and appendix map |
| 3 | 36 | Crypto-20 downstream alpha retest | Required before any multi-asset predictive claim | `crypto20_walkforward_experiment_results.csv`, `crypto20_walkforward_guided_alpha_comparison.csv`, `crypto20_alpha_generalization.md` |
| 4 | 37 | Crypto-50 or broader expansion | Conditional on Crypto-20 downstream evidence and compute budget | `crypto50_results.csv`, `generalization.md` |

## Minimal Ablation Definition

Phase 25 includes only the ablations needed to support or falsify the current paper mechanism:

| Ablation | Question |
|---|---|
| Vanilla contrastive vs HMM-guided contrastive | Does weak HMM supervision improve learned structure? |
| GMM vs HMM assignment on embeddings | Does sequential assignment matter after representation learning? |
| Time-only vs time-frequency guided encoder | Does the FFT view improve or dilute the guided representation? |
| Encoder depth if compute allows | Is the result sensitive to a minimal architecture change? |

The completed Phase 25 table supports the assignment-layer mechanism most strongly and does not justify expanding time-frequency downstream compute yet. Phase 26 preserves the paper-safe read: HMM assignment is the strongest supported mechanism, while guided-HMM versus raw-feature HMM remains directional rather than statistically decisive. Hard-negative mining remains optional unless the formal paper draft exposes a mechanism gap.

## Submission Readiness Checklist

The project is submission-ready only when:

1. Phase 24 protocol files are committed.
2. Phase 25 ablation artifacts exist and are discussed.
3. Phase 26 refreshed statistical evidence exists.
4. Phase 27 paper skeleton and artifact map exist.
5. Phase 28 reproduction commands and artifact policy docs exist.
6. Phase 29 prose pass preserves the claims allowed by `reports/claim_registry.md`.
7. The paper draft uses only claims allowed by `reports/claim_registry.md`.
8. The validation audit passes with no critical failures.
9. Reproduction commands are documented.


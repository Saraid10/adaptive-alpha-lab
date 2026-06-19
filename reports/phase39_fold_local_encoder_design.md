# Phase 39 Fully Fold-Local Encoder Design

## Purpose

This document is the implementation contract for Gate 1 in `reports/publication_acceptance_gates.md`. It maps the current offline `guided_encoder.py` and fold-local assignment logic in `walkforward_regimes.py` into a fully inductive outer-fold pipeline.

Phase 39 must not change Phase 36/37 artifacts. It writes new prefixed outputs and treats all existing Crypto-20 outcomes as development-observed.

## Implementation Status

Implemented on 2026-06-19. Seven targeted unit tests pass, and the corrected one-fold end-to-end smoke run completed in 459.6 seconds with exact coverage parity across all eight methods. The run used one epoch and at most 5,000 windows, so its metric values are deliberately excluded from scientific interpretation. The full development gate remains pending.

## Current Boundary Problem

`guided_encoder.py` currently fits one scaler across the loaded feature matrices, consumes a precomputed HMM assignment artifact, trains one encoder, and extracts one dense embedding matrix. `walkforward_regimes.py` then correctly refits GMM/HMM assignment layers inside each fold, but it consumes that previously trained dense embedding matrix.

The final learned-regime claim therefore needs a stricter boundary: the scaler, weak-supervision HMM, contrastive training pairs, and encoder weights must also be fit within each outer fold.

## Proposed Modules

### `src/fold_local_encoder.py`

Reusable pure components:

- `FoldEncoderConfig`: seed, method, augmentation, epochs, batch size, window size, pair-mining settings, and inner-validation settings.
- `FoldEncoderBounds`: outer-train, inner-train, inner-validation, embargo, purge, and outer-test timestamps.
- `FoldEncoderArtifacts`: fitted scaler, HMM, encoder weights, embedding metadata, loss history, and manifest values.
- `fit_training_scaler`: fit only on feature rows ending inside the supplied training boundary.
- `fit_training_hmm`: fit the weak-supervision HMM on scaled training rows with per-symbol sequence lengths.
- `build_training_windows`: construct windows whose start and end are both inside the authorized training interval.
- `train_vanilla_encoder`: train the vanilla contrastive reference.
- `train_guided_encoder`: train from fold-local HMM proxy states.
- `encode_causal_rows`: transform train or test endpoints with frozen scaler and encoder weights.
- `validate_fold_encoder_boundaries`: reject any unauthorized feature endpoint or pair.

The reusable encoder and loss classes remain in `encoder.py` and `guided_encoder.py`; training orchestration moves to fold-local functions rather than shelling out to the offline script.

### `src/fold_local_encoder_walkforward.py`

Outer-fold orchestrator:

1. Resolve the common universe and existing walk-forward fold boundaries.
2. Create an inner chronological validation block from the outer-training interval.
3. Select the epoch count inside the inner split.
4. Refit the scaler, weak-supervision HMM, and encoder on the full authorized outer-training interval using the selected epoch count.
5. Encode outer-training and outer-test endpoints causally.
6. Reuse fold-local GMM/HMM assignment, LightGBM training, prediction, and coverage validation.
7. Save compact fold manifests and summaries; keep weights and dense embeddings in ignored run storage.

## Inner Validation Contract

The initial implementation uses the final 720 eligible hourly endpoints of each outer-training period as inner validation, separated from inner training by the same label purge and a configurable inner embargo. The exact interval is recorded per fold.

Epoch selection may use validation contrastive loss, valid-anchor coverage, and a pre-specified structural diagnostic. It may not use outer-test IC, Sharpe, return, NLL, or any other outer-test result.

After selecting the epoch count, the fold encoder is refit once on the full authorized outer-training data. Phase 39 does not select architectures or feature families.

## Window And Causality Rules

- A training window is authorized only when every feature row in the window lies inside the training interval.
- A test embedding may use historical context available at or before its prediction timestamp, including embargo history and earlier test features, because the frozen encoder is only transforming observed information.
- A test embedding may not use a feature timestamp after its prediction endpoint.
- Weak-supervision HMM states are training-only labels; outer-test HMM states are not required to train the encoder.
- HMM filtering used as an assignment method must remain causal and carry only the terminal training posterior across the train-test gap.
- Row identity is keyed by `symbol`, `feat_idx`, and `open_time`; positional alignment alone is not accepted.

## Baseline Ladder

Phase 39 produces equal-coverage results for:

1. global LightGBM,
2. volatility buckets,
3. raw-feature KMeans,
4. raw-feature HMM,
5. fold-local vanilla contrastive-GMM,
6. fold-local vanilla contrastive-HMM,
7. fold-local HMM-guided contrastive-GMM,
8. fold-local HMM-guided contrastive-HMM.

The vanilla and guided encoders use the same outer folds, feature windows, latent dimension, training budget, and downstream evaluation unless the method definition itself requires a difference.

## Artifact Contract

Compact committed outputs:

- `models/crypto20_fold_local_encoder_manifest.csv`
- `models/crypto20_fold_local_encoder_loss.csv`
- `models/crypto20_fold_local_encoder_coverage.csv`
- `models/crypto20_fold_local_experiment_results.csv`
- `models/crypto20_fold_local_method_comparison.csv`
- `reports/phase39_fold_local_results.md`

Ignored heavy outputs:

- fold-specific model weights,
- dense train/test embeddings,
- row-level outer predictions,
- temporary pair-mining caches.

Every manifest row records fold, method, seed, timestamps, row counts, window counts, selected epoch, training epochs, device, input hashes, output hashes, and runtime.

## Required Tests

1. Scaler statistics change when and only when authorized training rows change.
2. Adding a future row cannot change an earlier fold scaler, HMM, pair set, or encoder input hash.
3. Every training window endpoint and start lies within the outer-training boundary.
4. Inner-validation rows never enter inner training.
5. Outer-test rows never enter pair mining or epoch selection.
6. Test embedding windows contain no timestamp after their prediction endpoint.
7. All eight methods have identical outer-test row coverage.
8. Repeated smoke runs with the same seed reproduce manifest hashes within the documented deterministic tolerance.

## Smoke And Full Commands

Planned smoke command:

```powershell
.\run_phase39_fold_local_encoder.ps1 -MaxFolds 1 -Epochs 1 -MaxWindows 5000
```

Planned full command:

```powershell
.\run_phase39_fold_local_encoder.ps1
```

## Compute Strategy

Phase 39 begins with one fold, one epoch, and capped windows. The full run is authorized only after boundary tests, coverage parity, deterministic manifests, and runtime estimates pass. Failed smoke gates are fixed before additional compute is spent.

The observed smoke runtime implies that the uncapped 16-fold, up-to-30-epoch run may require many hours on CPU. It must therefore be launched as a deliberate long-running development experiment with stable power, sufficient disk space, and no configuration changes prompted by smoke metrics. Early stopping uses inner-validation loss only. Partial or failed runs must not be silently presented as full evidence.

## Research Basis

The protocol follows three established ideas: selection on repeatedly inspected backtests creates overfitting risk; contrastive time-series representations must preserve temporal context; and evaluation must separate representation fitting from future observations. Relevant primary sources include Bailey et al., *The Probability of Backtest Overfitting* (Journal of Computational Finance, 2016, DOI `10.21314/JCF.2016.322`), van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (`arXiv:1807.03748`), and Yue et al., *TS2Vec: Towards Universal Representation of Time Series* (AAAI 2022, DOI `10.1609/aaai.v36i8.20881`). These sources motivate the separation and temporal controls; they do not validate this project's empirical claims.

## Phase 39 Exit Gate

Gate 1 and Gate 2 pass only when the full pipeline is demonstrably fold-local, all eight methods have equal outer-test coverage, no outer result influenced training decisions, and the validation audit can reconstruct the fold boundary evidence from saved manifests.

# Adaptive Alpha Lab Compute Budget

## Purpose

This document keeps the research plan compute-aware. The next phases add encoder upgrades and ablations on top of the frozen baseline, statistical tests, and regime-quality diagnostics. Without explicit compute gates, the project can grow into more experiments than a local machine can reasonably run.

## Current Environment

| Field | Value |
|---|---|
| Local Python | 3.11.3 |
| OS | Windows |
| Main research dependencies | PyTorch, LightGBM, hmmlearn, scikit-learn, DuckDB |
| Current universe | BTCUSDT + ETHUSDT |
| Current interval | 1h |
| Current feature rows | 34,920 total before target join |
| Current primary OOS rows | 25,920 per method |

The current run registry snapshot is:

```text
runs/20260522_phase14b_baseline/
```

## Observed Runtime Notes

These are practical local observations, not formal benchmarks:

| Task | Observed Runtime |
|---|---:|
| Phase 14A full symbol/horizon robustness grid | about 30 minutes |
| Phase 14B cost/threshold/period stress grid | under 1 minute |
| Phase 15A/15B statistical tests | under 1 minute |
| Phase 16 regime-quality diagnostics | under 1 minute |
| Phase 17 compute planner | about 20 seconds with 5 profiled CPU steps |
| Phase 18 one-epoch guided encoder smoke run | about 4 minutes |
| Validation audit | about 10 seconds |
| Data health check | a few seconds |

Phase 17 measured the current encoder training cost with a synthetic CPU forward/backward timing profile. This is not a formal hardware benchmark, but it is enough to keep the next experiment queue realistic.

| Metric | Value |
|---|---:|
| Device | CPU |
| Encoder parameters | 139,408 |
| Training windows | 34,798 |
| Batches per epoch | 271 |
| Synthetic step time | 0.734 seconds |
| Estimated epoch time | 3.32 minutes |
| Estimated 30-epoch encoder run | 99.45 minutes |
| Estimated 12-run ablation grid | 21.49 hours |
| Local budget | 24 hours |
| Budget status | green |

## Mandatory Next Experiments

The next phases should stay small until the baseline is statistically understood.

| Phase | Required Runs | Notes |
|---|---:|---|
| Phase 15A statistical tests | 0 retraining runs | Complete; uses frozen predictions/artifacts |
| Phase 15B multiple-testing corrections | 0 retraining runs | Complete; extends Phase 15A tables |
| Phase 16 regime quality metrics | 0 retraining runs | Complete; uses existing regime assignments |
| Phase 17 compute plan | 0 retraining runs | Complete; measured local encoder cost using synthetic timing |
| Phase 18 HMM-guided encoder | 1-2 encoder runs | Smoke-tested for 1 epoch; full 30-epoch run remains |
| Phase 19 time-frequency augmentation | 2-3 encoder runs | Time-only, frequency-only, both |
| Phase 20 hard negatives | 2-3 encoder runs | Random vs in-trajectory vs boundary negatives |
| Phase 21 ablations | capped matrix | Expand only if early results justify it |

## Initial Ablation Cap

Do not start with a full combinatorial grid. The initial cap should be:

```text
3 losses x 2 augmentations x 2 assignment methods = 12 runs maximum
```

Where:

```text
losses = NT-Xent, InfoNCE-style, HMM-guided
augmentations = time-only, time+frequency
assignment methods = GMM, HMM
```

Encoder depth and hidden-dimension ablations should only happen after the loss/augmentation/assignment choices show promise.

The first three queued runs are:

| Priority | Loss | Augmentation | Assignment | Decision |
|---:|---|---|---|---|
| 1 | `hmm_guided` | `time_only` | `hmm` | run first |
| 2 | `hmm_guided` | `time_only` | `gmm` | run first |
| 3 | `hmm_guided` | `time_frequency` | `hmm` | run first |

The rest of the 12-run grid should stay on hold until one of these runs improves the learned-regime path.

## Phase 18 Smoke Result

The first Phase 18 implementation run used `guided_encoder.py --epochs 1` to validate the full artifact path before launching the longer run. It produced both GMM and HMM assignments from the HMM-guided embedding space.

| Method | Epochs | Silhouette | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|
| `hmm_guided_gmm` | 1 | 0.341 | 0.387 | 0.652 |
| `hmm_guided_hmm` | 1 | 0.353 | 0.389 | 0.620 |

This is a smoke-test result, not a final performance claim. The full Phase 18 run should use 30 epochs before downstream alpha/statistical comparisons.

## Multi-Asset Gate

Phase 23 multi-asset generalization is conditional.

Proceed to multi-asset only if:

```text
best learned encoder beats raw-feature HMM on the primary predictive metric
and the paired/DM-style test reports p < 0.05.
```

Primary metric:

```text
signal IC or a loss-based predictive score from out-of-sample predictions
```

Secondary metrics:

```text
Sharpe, drawdown, total return, turnover
```

If the gate fails:

```text
Do not expand scope.
Write the paper as a crypto benchmark.
List multi-asset validation as future work.
```

This gate prevents scope creep and avoids spending compute on a generalization claim before the learned encoder has earned it.

## Paper-Safety Rules

- Every major run must be archived with `src/archive_run.py`.
- Every statistical table must reference a `run_id`.
- Every encoder result must record architecture, seed, loss, augmentation, assignment method, and source artifact paths.
- HMM states must be described as proxy/reference regimes, not ground truth.
- Backtest metrics must be described as diagnostics, not deployable trading claims.

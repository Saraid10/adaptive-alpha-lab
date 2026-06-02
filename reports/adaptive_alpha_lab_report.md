# Adaptive Alpha Lab Research Note

## Problem Statement

Financial markets are non-stationary: a signal that works in one market state can fail in another. Adaptive Alpha Lab tests whether learned market regimes improve alpha modeling compared with a global no-regime model and classical regime baselines under financial labels, purged walk-forward validation, and transaction costs.

The central question is:

> Do learned regimes improve IC, drawdown, Sharpe, or turnover versus global and classical-regime baselines?

## Related Work And Contribution Positioning

Phase 19A maps the project against four literature clusters:

1. Contrastive time-series representation learning: TS2Vec, TNC, CoST, TS-TCC, and TF-C show that unlabeled time-series windows can be embedded through carefully designed temporal, contextual, and frequency-aware objectives.
2. Classical financial regime switching: Hamilton-style Markov switching, Gaussian HMMs, and Markov-switching GARCH motivate sequential latent-state baselines with explicit transition structure.
3. Financial ML validation: triple-barrier labeling, purging, embargoing, fold-level tests, and multiple-testing controls define the evidence standard for financial prediction.
4. Regime-conditioned alpha modeling: predictors can be trained or weighted differently across market states, but the comparison must use the same labels, folds, costs, and test universe.

The paper contribution is therefore not a claim that HMM states are true regimes or that the project is a profitable trading system. The contribution is a controlled benchmark testing whether learned regime embeddings can beat, match, or explain classical sequential regimes under financial validation. Phase 18 then becomes a natural model-side response: use HMM states as weak supervision to improve contrastive representation learning, while still treating the HMM as a proxy rather than ground truth.

The narrative related-work note is `reports/related_work.md`; the compact source matrix is `reports/literature_matrix.csv`.

## Data And Features

The current benchmark uses hourly Binance OHLCV data for BTCUSDT and ETHUSDT. Each symbol has 17,520 OHLCV bars and 17,460 feature rows after indicator warmup. The feature store contains 22 engineered technical and microstructure-inspired features, including multi-horizon returns, realized volatility, volatility of volatility, Amihud illiquidity, volume z-score, return autocorrelation, spread proxy, order-flow proxy, RSI, Garman-Klass volatility, ATR, close-vs-VWAP, log volume trend, and return dispersion.

## Target Labeling

The primary modeling target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The neutral class captures periods where neither the profit barrier nor the stop barrier is hit before time expiry. Direction and volatility-adjusted labels are also generated at 4-hour, 8-hour, and 24-hour horizons.

The final target table contains 34,872 rows across BTCUSDT and ETHUSDT. For the primary 8-hour triple-barrier target, the neutral class is about 58.3% for both symbols, with up/down classes each near 20.6-21.0%.

## Regime Methods

The benchmark compares five regime methods on a common BTC+ETH universe:

| Method | Implementation | Rows | Silhouette | Avg Duration |
|---|---|---:|---:|---:|
| contrastive | contrastive encoder + GMM | 34,754 | 0.0959 | 30.51 |
| contrastive_hmm | contrastive embeddings + HMM | 34,754 | 0.1016 | 42.18 |
| hmm | hmmlearn Gaussian HMM | 34,754 | 0.0790 | 6.98 |
| kmeans | sklearn KMeans | 34,754 | 0.0967 | 4.05 |
| vol_bucket | realized-volatility quantiles | 34,754 | -0.0393 | 10.44 |

The contrastive method now uses dense stride-1 inference for every valid feature row after the 60-bar encoder window. This fixes the earlier sparse-coverage issue and makes downstream alpha comparisons fair.

Phase 11 adds a contrastive-HMM hybrid. Instead of clustering learned embeddings with GMM only, it fits a Gaussian HMM directly on the learned contrastive embedding sequence. This tests whether the weakness in contrastive regimes comes from representation learning itself or from the lack of temporal state dynamics in the assignment layer.

Phase 16 adds a separate structural regime-quality layer. It measures whether regimes are balanced, persistent, confident, and mutually consistent before evaluating whether they improve downstream alpha. The raw-feature HMM sequence is used as a classical reference proxy, not as ground truth.

## Regime Stability Diagnostics

Phase 10 adds explicit stability diagnostics to separate regime persistence from downstream usefulness. This matters because a regime method can look visually smooth while still being weak for alpha conditioning.

| Method | Switches / 1k Bars | Avg Duration | Transition Diagonal | Stable IC | Transition IC |
|---|---:|---:|---:|---:|---:|
| contrastive | 32.72 | 30.51 | 0.967 | -0.0200 | 0.0482 |
| contrastive_hmm | 23.65 | 42.18 | 0.976 | 0.0033 | 0.0100 |
| hmm | 143.24 | 6.98 | 0.857 | 0.0084 | -0.0061 |
| kmeans | 246.60 | 4.05 | 0.753 | 0.0083 | -0.0137 |
| vol_bucket | 95.71 | 10.44 | 0.904 | 0.0001 | -0.0168 |

The important finding is not simply that more stable regimes are better. Contrastive-HMM produces the longest-lived regimes and repairs the negative stable-period IC seen in contrastive-GMM. However, the raw-feature HMM still has the strongest stable-period IC. This suggests that HMM-style temporal structure helps learned embeddings, but the current embedding space is still not as alpha-aligned as the simpler raw-feature HMM state space.

## Validation Setup

Alpha models use expanding walk-forward validation with:

- 6-month initial training window
- 1-month test step
- 5-day embargo between train and test
- multiclass LightGBM over labels `-1`, `0`, `+1`
- transaction cost of 10 bps per trade

The global baseline trains one LightGBM model across both symbols. Regime-aware models train separate LightGBM classifiers per regime and combine predictions through posterior weights where available. The alpha score is `P(+1) - P(-1)`, and trades are taken only when the neutral class is not dominant and the score clears the threshold.

## Validation Audit

Phase 12 adds an explicit validation audit so the benchmark can be evaluated as research evidence rather than a collection of backtest claims. The audit checks database tables, feature/target schema, finite joined rows, target horizon tail loss, common benchmark coverage, fold separation, embargo spacing, label-horizon purging, row-level prediction alignment, duplicate predictions, and consistency between `alpha_oos_predictions.csv` and `experiment_results.csv`.

The audit result is:

| Status | Count | Interpretation |
|---|---:|---|
| PASS | 26 | All critical data, fold, target, coverage, prediction-alignment, Phase 20 fold-local artifact, robustness artifact, stress-grid, statistical-test artifact, regime-quality artifact, compute-plan artifact, guided-encoder full-run artifact, literature-positioning artifact, and run-registry checks passed |
| WARN | 1 | Legacy `regime_assignments.csv` is an offline/global artifact |
| FAIL | 0 | No critical validation failure was detected |

The most important positive result is that all 18 folds satisfy row separation, the 120-bar embargo, and the 8-bar primary label-horizon purge. The legacy offline alpha artifact still has equal coverage across six methods, while the Phase 20 fold-local artifact has equal coverage across eight methods, including the two guided-regime methods, with 25,920 rows each. The audit also confirms that the Phase 14A robustness matrix contains all 54 expected method/cell rows across 9 grid cells, that the Phase 14B stress matrix contains all 288 expected method/cell rows across 48 stress cells, that the Phase 15A/15B statistical artifacts are complete, that the Phase 16 regime-quality artifacts are complete, that the Phase 17 compute-plan artifacts are complete, that the Phase 19B guided-encoder full-run artifacts are complete, that the Phase 19A literature-positioning artifacts are complete, and that the frozen run registry points to a complete archived baseline.

The warning is methodological rather than a code failure: the legacy `regime_assignments.csv` file is generated as an offline/global artifact before alpha-model validation. This is acceptable for descriptive regime analysis and exploratory benchmarking. Phase 13 addresses the predictive version of this concern by adding a separate fold-local regime refit benchmark.

## Reproducibility Snapshot

Phase 15.0 adds artifact versioning before any statistical testing or encoder changes. The current frozen baseline is:

| Field | Value |
|---|---|
| Run id | `20260522_phase14b_baseline` |
| Source tag | `v1.3-phase14b` |
| Source commit | `f0902b21057d3dc464ce9e31d6be718a70531c63` |
| Manifest | `runs/20260522_phase14b_baseline/manifest.json` |
| Curated artifacts | 39 |
| Missing artifacts | 0 |

The `models/` directory remains the latest-artifact location used by the dashboard. The `runs/` directory stores immutable research snapshots with SHA-256 hashes, artifact manifests, git source references, and dependency files. Future statistical tests and encoder ablations should cite a `run_id` rather than relying on mutable latest CSV files.

## Fold-Local Regime Refit Benchmark

Phase 13 adds a stricter predictive benchmark. For each walk-forward fold, regime assignment models are refit using training-history rows only and are then applied to the test window. HMM-based methods use an online filtering pass for test assignments, initialized from the training sequence and advanced through the embargo gap. This avoids using a single full-sample regime assignment file for predictive alpha claims.

Phase 20 extends this same strict benchmark with two HMM-guided learned-regime methods. The guided embeddings are frozen from the Phase 19B encoder run, but the GMM/HMM assignment layer on top of those embeddings is fit fold-locally.

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0110 | 0.5623 | 0.3708 | -0.834 | -0.926 | 0.074 | -0.823 |
| regime_lgbm_contrastive_hmm | -0.0026 | 0.5623 | 0.3730 | -0.548 | -0.778 | 0.077 | -0.685 |
| regime_lgbm_hmm | 0.0051 | 0.5623 | 0.3698 | -0.340 | -0.710 | 0.079 | -0.536 |
| regime_lgbm_kmeans | 0.0072 | 0.5672 | 0.3704 | -0.728 | -0.860 | 0.081 | -0.797 |
| regime_lgbm_vol_bucket | -0.0020 | 0.5570 | 0.3679 | -0.820 | -0.854 | 0.083 | -0.820 |
| regime_lgbm_hmm_guided_gmm | -0.0092 | 0.5600 | 0.3669 | -0.976 | -0.900 | 0.079 | -0.854 |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.5623 | 0.3743 | 0.099 | -0.614 | 0.084 | 0.031 |

The Phase 20 benchmark changes the interpretation. The original fold-local learned-regime path remains weak, and guided embeddings with GMM also fail. However, guided embeddings with an HMM assignment layer become the strongest point-estimate method: they beat raw-feature HMM by `+0.0043` IC, `+0.439` Sharpe, `+0.095` drawdown, and `+0.567` total return. This supports a precise mechanism: HMM-guided representation learning helps only when the downstream assignment layer also enforces sequential state dynamics.

## Phase 14A Robustness Matrix

Phase 14A tests whether the fold-local result is stable across assets and horizons. The same benchmark is repeated across BTC-only, ETH-only, and BTC+ETH scopes at 4-hour, 8-hour, and 24-hour triple-barrier horizons. Regime assignment models are still refit inside each fold; the matrix changes the target horizon and symbol scope, not the validation discipline.

| Scope | Target | Best IC Method | Best IC | Best Sharpe Method | Best Sharpe | Lowest Drawdown Method | Lowest Drawdown |
|---|---|---|---:|---|---:|---|---:|
| BTCUSDT | tb_label_4h | regime_lgbm_hmm | 0.0016 | regime_lgbm_hmm | -0.993 | regime_lgbm_contrastive | -0.755 |
| BTCUSDT | tb_label_8h | regime_lgbm_kmeans | -0.0034 | regime_lgbm_contrastive | -0.546 | regime_lgbm_contrastive | -0.812 |
| BTCUSDT | tb_label_24h | regime_lgbm_kmeans | 0.0175 | regime_lgbm_contrastive_hmm | -0.211 | regime_lgbm_kmeans | -0.990 |
| ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0208 | regime_lgbm_vol_bucket | 0.315 | regime_lgbm_vol_bucket | -0.449 |
| ETHUSDT | tb_label_8h | global_lgbm | 0.0095 | regime_lgbm_contrastive | -0.201 | regime_lgbm_hmm | -0.874 |
| ETHUSDT | tb_label_24h | global_lgbm | 0.0348 | global_lgbm | 0.354 | regime_lgbm_vol_bucket | -0.987 |
| BTCUSDT+ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0103 | regime_lgbm_hmm | -0.205 | regime_lgbm_hmm | -0.436 |
| BTCUSDT+ETHUSDT | tb_label_8h | regime_lgbm_kmeans | 0.0072 | regime_lgbm_hmm | -0.340 | global_lgbm | -0.688 |
| BTCUSDT+ETHUSDT | tb_label_24h | regime_lgbm_contrastive_hmm | 0.0311 | regime_lgbm_vol_bucket | 0.321 | regime_lgbm_vol_bucket | -0.915 |

The robustness matrix weakens any simple claim that one regime method is universally best. KMeans wins IC most often, HMM wins Sharpe most often, and volatility buckets win drawdown most often. The contrastive-HMM hybrid wins the BTC+ETH 24-hour IC cell, which is useful but not enough to claim learned regimes dominate globally.

The research conclusion becomes more precise: regime conditioning can help, but the choice of regime model is metric-, horizon-, and asset-dependent. This is a stronger and more honest result than a single headline Sharpe because it shows where each method is fragile.

## Phase 14B Stress Robustness

Phase 14B tests whether the fold-local `tb_label_8h` conclusion survives practical trading assumptions. Instead of retraining models, it re-scores the same out-of-sample prediction file across transaction costs, signal thresholds, and market-period slices.

| Stress Dimension | Values |
|---|---|
| Signal threshold | 0.03, 0.05, 0.07, 0.10 |
| Transaction cost | 5 bps, 10 bps, 20 bps |
| Market period | all, bull, sideways, bear |

Bull, sideways, and bear periods are defined from rolling 30-day returns in the feature store. This creates 48 stress cells and 288 method/cell rows.

| Metric | Most Frequent Winner | Wins |
|---|---|---:|
| Signal IC | regime_lgbm_hmm | 24 |
| Sharpe | regime_lgbm_hmm | 22 |
| Drawdown | global_lgbm | 24 |
| Total return | regime_lgbm_hmm | 18 |

The stress grid strengthens the HMM interpretation. Raw-feature HMM wins the most signal-IC, Sharpe, and total-return cells across cost, threshold, and market-period settings. The global model is the most defensive drawdown winner, which is expected because fewer regime-conditioned switches can reduce downside under higher costs. Contrastive-HMM remains useful in sideways regimes, but it is not the dominant learned-regime method.

The important result is not that a strategy is profitable. The important result is that the relative conclusion is stress-tested: HMM-style temporal state structure is the most robust regime-aware layer in the current implementation, while learned embeddings still need a stronger objective or stricter fold-local representation training to dominate.

## Phase 15A/15B Statistical Rigor

Phase 15A tests whether the fold-local method differences are statistically reliable. The primary unit for IC and Sharpe significance is the walk-forward fold, not individual rows, because adjacent hourly labels overlap and row-level samples are not independent. For calibration-oriented forecast quality, the phase also runs a Newey-West DM-style test on per-row multiclass negative log-likelihood.

Phase 15B then applies multiple-testing controls. It uses Benjamini-Hochberg false-discovery-rate correction and Holm family-wise correction so that a single attractive raw p-value is not treated as a publishable claim after many method/metric comparisons.

| Method | Mean Fold IC | 95% CI Low | 95% CI High | Positive IC Folds | Mean Fold Sharpe |
|---|---:|---:|---:|---:|---:|
| regime_lgbm_hmm_guided_hmm | 0.0080 | -0.0122 | 0.0278 | 9 | -0.026 |
| regime_lgbm_hmm | 0.0058 | -0.0135 | 0.0247 | 11 | -0.561 |
| regime_lgbm_kmeans | 0.0035 | -0.0202 | 0.0282 | 8 | -0.720 |
| regime_lgbm_vol_bucket | 0.0004 | -0.0230 | 0.0241 | 10 | -0.818 |
| global_lgbm | -0.0005 | -0.0207 | 0.0209 | 9 | -0.583 |
| regime_lgbm_contrastive_hmm | -0.0063 | -0.0305 | 0.0196 | 7 | -0.908 |
| regime_lgbm_hmm_guided_gmm | -0.0075 | -0.0336 | 0.0189 | 8 | -1.058 |
| regime_lgbm_contrastive | -0.0147 | -0.0373 | 0.0095 | 7 | -0.990 |

The confidence intervals are wide and mostly overlap zero. That means the project should not claim that the current guided-HMM edge is statistically decisive from IC alone. The strongest point-estimate result is positive: `regime_lgbm_hmm_guided_hmm` has the highest mean fold IC and the least negative mean fold Sharpe. Against raw-feature HMM, however, the IC improvement is small relative to fold noise (`mean difference = 0.0022`, paired fold `p = 0.801`). The strongest fold-level significant result remains a negative finding: contrastive-GMM is worse than raw-feature HMM on IC (`mean difference = -0.0205`, paired fold `p = 0.035`). After Phase 15B correction, this becomes `raw_only_suggestive` rather than a hard corrected claim. Most other IC and Sharpe differences are not significant at the 5% level.

The row-level DM-style negative-log-likelihood tests show that the global model is often better calibrated than the regime-conditioned models. This does not contradict the IC result; it separates directional/ranking usefulness from probability calibration. For a paper, this is valuable because it prevents overclaiming: the regime methods may improve some alpha diagnostics, but their probability estimates are not automatically better calibrated.

Probabilistic Sharpe Ratio diagnostics improve materially after Phase 20: `regime_lgbm_hmm_guided_hmm` has `PSR(SR > 0) = 0.633`, while raw-feature HMM is about `0.121`. This is useful support for the guided-HMM direction, but it remains a diagnostic on overlapping portfolio returns, not a deployable performance claim.

## Phase 16 Regime Quality

Phase 16 evaluates regime methods structurally before asking whether they help trading. This is important because a regime method can look smooth or produce a better backtest by chance without actually discovering meaningful state partitions.

The metrics include:

- regime balance entropy: whether one regime dominates all rows
- switch rate and transition diagonal probability: how persistent the state sequence is
- posterior confidence and entropy: how uncertain soft-assignment methods are
- pairwise NMI/ARI: how much two regime methods agree after ignoring arbitrary label names
- HMM-reference NMI/purity: agreement with the classical raw-feature HMM proxy

| Method | Balance Entropy | Switches / 1k Bars | Transition Diagonal | Avg Duration | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| contrastive | 0.999 | 32.72 | 0.967 | 30.51 | 0.032 | 0.379 |
| contrastive_hmm | 0.999 | 23.65 | 0.976 | 42.18 | 0.020 | 0.377 |
| hmm | 0.959 | 143.24 | 0.857 | 6.98 | 1.000 | 1.000 |
| kmeans | 0.866 | 246.60 | 0.753 | 4.05 | 0.182 | 0.459 |
| vol_bucket | 1.000 | 95.71 | 0.904 | 10.44 | 0.333 | 0.599 |

The main finding is that learned regimes are structurally smooth but not strongly aligned with the HMM reference. Contrastive-HMM has the longest average duration and highest transition diagonal, but its HMM-reference NMI is only 0.020. Volatility buckets have the strongest non-HMM agreement with the HMM reference (`NMI = 0.333`, purity `0.599`), even though the method is much simpler.

This supports the paper's core diagnostic interpretation: persistence alone is not enough. The learned encoder currently creates stable partitions, but those partitions are not yet aligned with the state structure that helps the downstream alpha benchmark. The next encoder phase should therefore change the representation-learning objective, not merely increase model size.

## Phase 17 Compute Plan

Phase 17 makes the encoder-upgrade roadmap compute-aware. Before launching HMM-guided losses, time-frequency views, or broader ablation grids, the project now records a local training-cost estimate and a capped experiment queue.

The current profile uses synthetic forward/backward timing for the existing `TemporalEncoder` on the local CPU-only environment.

| Metric | Value |
|---|---:|
| Encoder parameters | 139,408 |
| Training windows | 34,798 |
| Batches per epoch | 271 |
| Profiled step time | 0.734 seconds |
| Estimated epoch time | 3.32 minutes |
| Estimated 30-epoch retrain | 99.45 minutes |
| Estimated 12-run initial grid | 21.49 hours |
| Local budget | 24 hours |
| Budget status | green |

The initial ablation cap remains 12 runs: 3 losses by 2 augmentations by 2 assignment methods. Phase 17 marks only three runs as `run_first`:

| Priority | Loss | Augmentation | Assignment |
|---:|---|---|---|
| 1 | `hmm_guided` | `time_only` | `hmm` |
| 2 | `hmm_guided` | `time_only` | `gmm` |
| 3 | `hmm_guided` | `time_frequency` | `hmm` |

This matters for paper execution because it prevents an uncontrolled ablation explosion. The project should first test whether HMM-guided supervision improves the learned-regime path. Only if the priority runs improve the fold-local learned-regime benchmark should the remaining grid be launched.

## Phase 18 HMM-Guided Encoder

Phase 18 implements the first encoder-objective upgrade. The current contrastive encoder treats adjacent windows as positives, which can encourage local smoothness even when the market is crossing a regime boundary. The Phase 18 encoder instead uses the raw-feature HMM sequence as weak supervision:

- same HMM state and distant in time: positive pair
- different HMM state and nearby in the same symbol: hard negative pair
- HMM is treated as a proxy/reference sequence, not market-regime ground truth

The implementation writes separate guided artifacts and does not overwrite the existing canonical encoder or benchmark assignments. This keeps the experiment reversible and prevents Phase 18 from contaminating the Phase 15/16 baseline.

The first run is intentionally a one-epoch smoke test to validate the training and artifact path before spending compute on a 30-epoch run.

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| `hmm_guided_gmm` | 1 | 0.341 | 15.17 | 0.934 | 0.387 | 0.652 |
| `hmm_guided_hmm` | 1 | 0.353 | 18.55 | 0.946 | 0.389 | 0.620 |

This is already directionally useful: the Phase 16 contrastive-GMM method had only `HMM NMI = 0.032`, while the one-epoch HMM-guided variant reaches about `0.39`. That does not prove alpha improvement, but it confirms that the guided objective changes the embedding geometry in the intended direction. The next step is a full 30-epoch guided run followed by fold-local alpha and statistical re-testing.

## Phase 19A Literature Positioning

Phase 19A adds a paper-facing related-work layer before expanding the encoder roadmap. This phase does not change model outputs. It changes the research framing so later experiments answer a precise question instead of becoming a collection of model variants.

The project is positioned against:

| Cluster | Key References | Why It Matters Here |
|---|---|---|
| Time-series contrastive learning | TS2Vec, TNC, CoST, TS-TCC, TF-C | Defines the representation-learning family that the encoder belongs to |
| Financial regime switching | Hamilton 1989, Gaussian HMMs, Markov-switching GARCH | Defines the classical sequential-state baseline the learned regimes must beat |
| Financial ML validation | triple-barrier labels, purging, embargoing, multiple-testing discipline | Defines what counts as credible evidence in a financial prediction benchmark |
| Regime-conditioned alpha | global versus state-conditioned predictors | Defines the downstream use case and evaluation surface |

This positioning clarifies the contribution statement: the project is an empirical bridge between classical sequential regime models and deep time-series representation learning. Its strongest current finding is not that deep regimes dominate, but that sequential state discipline matters and that HMM-guided weak supervision may repair part of the contrastive encoder's regime-structure weakness.

## Phase 19B Full Guided Encoder Run

Phase 19B runs the HMM-guided encoder for the planned 30 epochs. This converts the Phase 18 smoke test into a real structural representation experiment. The training loss falls steadily from `0.9284` at epoch 1 to `0.0949` at epoch 30, while the valid-anchor percentage remains `1.000`.

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | Mean Confidence | HMM NMI | HMM ARI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmm_guided_gmm` | 30 | 0.384 | 5.09 | 0.804 | 0.991 | 0.609 | 0.434 | 0.759 |
| `hmm_guided_hmm` | 30 | 0.629 | 5.72 | 0.825 | 0.998 | 0.869 | 0.887 | 0.957 |

The comparison artifact shows the structural jump clearly:

| Method | Source | HMM NMI | HMM Purity | NMI Delta vs Contrastive |
|---|---|---:|---:|---:|
| `contrastive` | Phase 16 | 0.032 | 0.379 | 0.000 |
| `hmm_guided_gmm` | Phase 19B | 0.609 | 0.759 | 0.577 |
| `hmm_guided_hmm` | Phase 19B | 0.869 | 0.957 | 0.837 |

This is the first strong model-side evidence that the weakly supervised objective is doing what it was designed to do. The learned representation path no longer merely creates smooth partitions; when paired with an HMM assignment layer, it becomes strongly aligned with the classical sequential reference. Phase 20 then tests whether that structural improvement survives contact with the downstream alpha benchmark.

## Phase 20 Guided Alpha Retest

Phase 20 adds the paper-critical downstream test for the Phase 19B encoder. The guided embeddings are kept frozen, but the regime assignment layer on top of them is refit inside each walk-forward fold. This avoids the invalid shortcut of using one full-sample guided assignment file for predictive claims.

Two learned-regime variants are evaluated:

1. `hmm_guided_gmm`: fold-local GMM assignment on HMM-guided embeddings.
2. `hmm_guided_hmm`: fold-local online HMM assignment on HMM-guided embeddings.

Both variants are evaluated on the same 25,920 out-of-sample rows as every other method.

## Model Comparison

| Method | IC | Accuracy | Balanced Accuracy | Sharpe | Drawdown | Turnover | Total Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_lgbm | 0.0024 | 0.5736 | 0.3625 | -0.506 | -0.688 | 0.050 | -0.557 |
| regime_lgbm_contrastive | -0.0110 | 0.5623 | 0.3708 | -0.834 | -0.926 | 0.074 | -0.823 |
| regime_lgbm_contrastive_hmm | -0.0026 | 0.5623 | 0.3730 | -0.548 | -0.778 | 0.077 | -0.685 |
| regime_lgbm_hmm | 0.0051 | 0.5623 | 0.3698 | -0.340 | -0.710 | 0.079 | -0.536 |
| regime_lgbm_kmeans | 0.0072 | 0.5672 | 0.3704 | -0.728 | -0.860 | 0.081 | -0.797 |
| regime_lgbm_vol_bucket | -0.0020 | 0.5570 | 0.3679 | -0.820 | -0.854 | 0.083 | -0.820 |
| regime_lgbm_hmm_guided_gmm | -0.0092 | 0.5600 | 0.3669 | -0.976 | -0.900 | 0.079 | -0.854 |
| regime_lgbm_hmm_guided_hmm | 0.0094 | 0.5623 | 0.3743 | 0.099 | -0.614 | 0.084 | 0.031 |

All methods are evaluated on 25,920 out-of-sample rows. This equal test coverage is important: it prevents a regime method from looking better simply because it was tested on a smaller or easier subset.

## Results Interpretation

Phase 20 changes the headline result. The strongest point-estimate method is now `regime_lgbm_hmm_guided_hmm`, with IC `0.0094`, Sharpe `0.099`, drawdown `-0.614`, and total return `0.031`. It beats the raw-feature HMM on the main point estimates while using the same walk-forward folds, costs, target, and test rows.

The mechanism is specific. Guided embeddings with GMM assignment still fail, while guided embeddings with HMM assignment become the best point-estimate method. That supports the main research thesis: representation learning helps only when the assignment layer preserves sequential state dynamics. Rich embeddings alone are not enough.

The statistical interpretation stays cautious. At the fold level, the guided-HMM IC advantage over raw-feature HMM is positive but not significant at the 5% level (`p = 0.801`). The guided-HMM method also has a stronger Probabilistic Sharpe read than raw HMM (`PSR(SR>0) = 0.633` versus about `0.121`), but calibration/NLL diagnostics still favor simpler references in some comparisons. The paper claim should therefore be phrased as a promising, controlled empirical finding rather than proof of a dominant trading strategy.

The project now has a clean scientific progression: dense contrastive regimes underperform, stability diagnostics identify the assignment-layer weakness, contrastive-HMM partly fixes temporal consistency, HMM-guided weak supervision repairs structural alignment, and Phase 20 shows that the guided-HMM path can become the strongest downstream point-estimate method under strict fold-local validation.

## Limitations

- Hourly OHLCV is a noisy signal source.
- The current contrastive encoder is not a true Temporal Fusion Transformer.
- The contrastive encoder is still trained as an offline/frozen representation; a future paper-grade upgrade is fold-local encoder retraining.
- Offline/global regime results and fold-local regime results should be interpreted separately.
- HMM states are not ground truth market regimes. They are a classical proxy/reference state sequence used for comparison and weak supervision.
- Regime-quality agreement metrics are diagnostic; agreement with HMM does not prove economic correctness.
- Phase 20 tests guided embeddings downstream, but the guided encoder itself is still trained as a frozen/offline representation. A stricter future version would retrain or update the encoder inside each walk-forward fold.
- The Phase 20 guided-HMM edge over raw-feature HMM is promising but not statistically significant at the 5% level on fold-level IC.
- Guided-HMM improves IC, Sharpe, drawdown, and total return point estimates, but the guided methods still need robustness-grid and stress-grid coverage before any broad robustness claim.
- Calibration/NLL diagnostics do not uniformly favor the guided methods, so probability quality and trading-score quality should be discussed separately.
- Phase 19A is a positioning phase, not an empirical result. It clarifies contribution language but does not prove a new model improvement.
- Literature positioning depends on describing HMM states as proxy/reference states, not ground truth labels.
- Phase 14B stress testing re-scores existing predictions; it does not retrain models under each cost or threshold assumption.
- Several method differences are not statistically significant at the 5% level under fold-level tests, and the strongest IC finding does not survive multiple-testing correction.
- Phase 17 timings are synthetic planning estimates, not formal hardware benchmarks.
- Backtest returns are research diagnostics, not deployable trading evidence.
- The project intentionally excludes live trading, RL, online retraining, and order-book data in this phase.

## Next Steps

1. Extend Phase 20 guided methods into robustness and stress testing before making a broad robustness claim.
2. Add feature importance and SHAP summaries for global, raw-HMM, and guided-HMM LightGBM models.
3. Add time-frequency augmentation and hard-negative ablations, but keep the grid capped by the compute budget.
4. Add fold-local or expanding-window encoder retraining for the learned-regime methods.
5. Use the Phase 19A literature matrix to write the formal related-work section before paper drafting.
6. Treat multi-asset expansion as conditional on statistically reliable learned-encoder improvement.

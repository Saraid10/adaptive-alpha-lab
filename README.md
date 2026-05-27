# Adaptive Alpha Lab

> A research-grade quant ML benchmark platform that tests whether learned market regimes improve alpha modeling versus global baselines and classical regime methods, using proper financial labels, purged walk-forward validation, and transaction-cost-aware evaluation.

**Live Demo:** [adaptive-alpha-engine.streamlit.app](https://adaptive-alpha-engine.streamlit.app/)

## Key Finding

The strongest current result is not that learned regimes beat every baseline. They do not. The true Gaussian HMM baseline remains the best IC and Sharpe method in the latest BTC+ETH run.

Phase 11 tested the natural research response: fit an HMM on learned contrastive embeddings. The contrastive-HMM hybrid improves the learned-regime path from `IC=-0.0165` to `IC=0.0035` and from `Sharpe=-0.902` to `Sharpe=-0.382`, but it still does not beat the raw-feature HMM. That is the current project insight: HMM-style temporal state structure helps learned embeddings, but representation learning alone is not yet enough.

## Research Question

Does learning market regimes from raw financial time-series features improve alpha-model IC, drawdown, and Sharpe compared with:

- a global model with no regime awareness
- Gaussian HMM regimes
- contrastive-HMM hybrid regimes
- KMeans regimes
- volatility-bucket regimes

The project is intentionally framed as a research benchmark, not a live trading bot. Honest weak or mixed results are part of the contribution.

## Research Positioning

Adaptive Alpha Lab sits between four research areas:

- contrastive representation learning for time series, such as TS2Vec, TNC, CoST, TS-TCC, and TF-C
- classical financial regime-switching models, including Hamilton-style Markov switching, Gaussian HMMs, and Markov-switching GARCH
- financial ML validation discipline: triple-barrier labels, purging, embargoes, fold-level testing, and multiple-testing control
- regime-conditioned alpha modeling, where predictors are trained or weighted differently across market states

The contribution is not "a new trading bot" or "HMM is ground truth." The contribution is a reproducible benchmark that tests whether learned regime embeddings can beat, match, or explain classical sequential regimes under financial validation. Phase 18 adds the first model-side response to the current finding: an HMM-guided contrastive encoder that uses HMM states as weak supervision rather than hard truth.

The full positioning note is in `reports/related_work.md`, with a compact source matrix in `reports/literature_matrix.csv`.

## Architecture

```text
Binance OHLCV
    -> DuckDB feature store
    -> 22 engineered market/microstructure features
    -> Multi-horizon financial labels and triple-barrier targets
    -> Contrastive temporal regime encoder + contrastive-HMM hybrid + classical baselines
    -> Global and regime-conditioned LightGBM models
    -> Purged walk-forward validation with transaction costs
    -> Research dashboard and report artifacts
```

## Current Capabilities

- Incremental Binance OHLCV ingestion for BTCUSDT and ETHUSDT.
- DuckDB feature store with 22 engineered features.
- Contrastive Transformer-style temporal encoder trained with NT-Xent loss.
- GMM regime posteriors and UMAP/timeline visualizations.
- Multi-horizon targets: 4h, 8h, and 24h.
- Triple-barrier labels with neutral class.
- Regime baselines: contrastive, Gaussian HMM, KMeans, volatility buckets.
- Contrastive-HMM hybrid regimes fitted on learned embeddings.
- Regime stability diagnostics: switch rate, duration, transition entropy, and stable-vs-transition IC.
- Phase 16 regime-quality diagnostics: balance, persistence, posterior confidence, pairwise NMI/ARI, and HMM-reference agreement.
- Global LightGBM baseline and regime-conditioned LightGBM models.
- 5-day embargoed walk-forward validation.
- Validation audit for fold separation, target horizon leakage, coverage parity, and prediction alignment.
- Fold-local regime refitting benchmark for stricter predictive regime evaluation.
- Phase 14A robustness matrix across BTC-only, ETH-only, and BTC+ETH scopes at 4h, 8h, and 24h horizons.
- Phase 14B stress robustness across transaction costs, signal thresholds, and bull/sideways/bear market periods.
- Phase 15.0 run registry for timestamped artifact snapshots and frozen baselines.
- Phase 15A statistical testing with fold-level bootstrap confidence intervals, paired tests, and DM-style forecast-loss checks.
- Phase 15B multiple-testing correction, corrected claim status, and Probabilistic Sharpe diagnostics.
- Phase 17 compute planning with encoder timing, ablation budget, and experiment-priority queue.
- Phase 18 HMM-guided contrastive encoder prototype with weak HMM-state positives and boundary-aware hard negatives.
- Phase 19A literature positioning across time-series contrastive learning, financial regime switching, financial ML validation, and regime-conditioned alpha modeling.
- Phase 19B full 30-epoch HMM-guided encoder run with guided-vs-baseline structural comparison.
- Transaction-cost-aware experiment result table.
- Streamlit dashboard shell and research note.

## Run Order

Recommended local environment:

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
```

```powershell
python src/check.py
python src/targets.py --symbols BTCUSDT ETHUSDT
python src/train_encoder.py --symbols BTCUSDT ETHUSDT
python src/visualize_regimes.py --symbols BTCUSDT ETHUSDT
python src/baselines.py --symbols BTCUSDT ETHUSDT
python src/alpha_models.py --symbols BTCUSDT ETHUSDT
python src/regime_stability.py --symbols BTCUSDT ETHUSDT
python src/regime_quality.py --symbols BTCUSDT ETHUSDT
python src/compute_plan.py --symbols BTCUSDT ETHUSDT
python src/guided_encoder.py --symbols BTCUSDT ETHUSDT --epochs 30
python src/walkforward_regimes.py --symbols BTCUSDT ETHUSDT
python src/robustness.py
python src/robustness_stress.py
python src/statistical_tests.py
python src/validation_audit.py --symbols BTCUSDT ETHUSDT
python src/archive_run.py --phase phase14b_baseline --run-id 20260522_phase14b_baseline --source-ref v1.3-phase14b --notes "Frozen Phase 14B baseline before Phase 15 statistical and encoder work."
python src/backtest.py
python -m compileall src dashboard.py streamlit_app.py
```

Optional dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

For Streamlit Cloud deployment, the root `requirements.txt` is intentionally minimal and installs only dashboard dependencies.

If local research dependencies are not installed, run:

```powershell
python -m pip install -r requirements-research.txt
```

## Key Artifacts

| Artifact | Purpose |
|---|---|
| `models/target_distribution.csv` | Label balance across horizons and label types |
| `models/target_quality.csv` | Missing-row, horizon-loss, and neutral-class checks |
| `models/regime_assignments.csv` | Aligned regime labels/posteriors for all methods |
| `models/regime_benchmark_summary.csv` | Regime-level comparison table |
| `models/regime_stability_summary.csv` | Persistence, switch-rate, confidence, and stable-vs-transition IC diagnostics |
| `models/regime_quality_summary.csv` | Phase 16 structural regime-quality metrics independent of alpha performance |
| `models/regime_agreement_matrix.csv` | Pairwise NMI/ARI/same-label agreement across regime methods |
| `models/regime_quality_heatmap.png` | Visual summary of balance, persistence, confidence, and HMM-reference agreement |
| `models/regime_agreement_heatmap.png` | Pairwise method-agreement heatmap |
| `models/compute_profile.csv` | Phase 17 local encoder timing and train-cost estimate |
| `models/ablation_budget.csv` | Prioritized 12-run encoder ablation queue |
| `models/compute_budget_summary.csv` | Compact compute-budget summary for dashboard/report use |
| `models/compute_budget_plan.png` | Visual ablation-runtime budget |
| `models/guided_encoder_summary.csv` | Phase 19B HMM-guided encoder structural diagnostics |
| `models/guided_encoder_loss.csv` | Phase 19B guided training loss and pair-mining diagnostics |
| `models/guided_encoder_comparison.csv` | Phase 19B comparison of guided encoder regimes versus existing structural baselines |
| `models/guided_encoder_loss_curve.png` | Visual Phase 19B training loss curve |
| `models/per_regime_stats.csv` | Volatility, return, liquidity, and IC diagnostics by regime |
| `models/experiment_results.csv` | Master model comparison table |
| `models/alpha_oos_predictions.csv` | Out-of-sample model predictions |
| `models/validation_audit.csv` | Leakage, embargo, fold, coverage, and prediction-alignment audit |
| `models/fold_audit.csv` | Fold-level train/test boundary and embargo checks |
| `models/walkforward_experiment_results.csv` | Strict fold-local regime-refit alpha benchmark |
| `models/walkforward_comparison.csv` | Offline/global versus fold-local regime result comparison |
| `models/walkforward_regime_summary.csv` | Fold-local regime coverage, duration, and confidence diagnostics |
| `models/robustness_results.csv` | Phase 14A full horizon/symbol-scope robustness matrix |
| `models/robustness_summary.csv` | Best method per robustness grid cell |
| `models/robustness_wins.csv` | Win counts by metric across robustness grid cells |
| `models/robustness_heatmap.png` | Visual summary of robustness winners |
| `models/robustness_stress_results.csv` | Phase 14B full cost/threshold/market-period stress grid |
| `models/robustness_stress_summary.csv` | Best method per stress grid cell |
| `models/robustness_stress_wins.csv` | Stress-test win counts by metric |
| `models/robustness_stress_heatmap.png` | Visual summary of stress-test sensitivity |
| `models/statistical_method_summary.csv` | Phase 15A fold-level confidence intervals by method |
| `models/statistical_pairwise_tests.csv` | Phase 15A paired and DM-style method comparisons |
| `models/statistical_test_summary.csv` | Compact paper-facing significance summary |
| `models/statistical_ic_confidence_intervals.png` | Fold-level IC confidence interval plot |
| `models/statistical_multiple_testing.csv` | Phase 15B corrected p-values across tested claims |
| `models/statistical_claims.csv` | Phase 15B corrected claim-status summary |
| `models/statistical_sharpe_diagnostics.csv` | Probabilistic Sharpe Ratio diagnostics |
| `models/statistical_multiple_testing.png` | Visual multiple-testing correction summary |
| `models/statistical_sharpe_diagnostics.png` | Visual PSR diagnostic summary |
| `runs/run_index.csv` | Versioned run registry |
| `runs/20260522_phase14b_baseline/manifest.json` | Frozen Phase 14B baseline manifest |
| `reports/model_card.md` | Reproducible model-card snapshot |
| `reports/compute_budget.md` | Compute-aware experiment plan and multi-asset gate |
| `reports/related_work.md` | Phase 19A paper-positioning note and contribution map |
| `reports/literature_matrix.csv` | Compact source matrix for paper planning |
| `models/regime_stability.png` | Stability and transition-period IC dashboard panel |
| `models/phase4_dashboard.png` | Static research backtest dashboard |
| `reports/adaptive_alpha_lab_report.md` | Research note |

## Project Structure

```text
adaptive-alpha-lab/
├── dashboard.py
├── streamlit_app.py
├── reports/
│   ├── adaptive_alpha_lab_report.md
│   ├── related_work.md
│   ├── literature_matrix.csv
│   ├── model_card.md
│   └── compute_budget.md
├── src/
│   ├── ingestion.py
│   ├── features.py
│   ├── targets.py
│   ├── dataset.py
│   ├── encoder.py
│   ├── train_encoder.py
│   ├── visualize_regimes.py
│   ├── baselines.py
│   ├── regime_stability.py
│   ├── regime_quality.py
│   ├── compute_plan.py
│   ├── guided_encoder.py
│   ├── validation_audit.py
│   ├── walkforward_regimes.py
│   ├── robustness.py
│   ├── robustness_stress.py
│   ├── statistical_tests.py
│   ├── archive_run.py
│   ├── alpha_models.py
│   ├── backtest.py
│   └── check.py
├── models/
│   └── generated research artifacts
└── runs/
    └── versioned curated run snapshots
```

## Methodology Notes

The primary target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The alpha score is `P(+1) - P(-1)`. A prediction becomes a trade only when the neutral class is not dominant and the absolute score exceeds the threshold.

The validation scheme uses expanding walk-forward folds with a 5-day embargo gap between train and test windows. This reduces leakage from overlapping financial labels and makes the model comparison more defensible.

Phase 12 adds a validation audit. The current audit passes all critical checks: required tables, feature/target schema, finite joined rows, 24-row target horizon loss, 18 walk-forward folds, 120-bar embargo, 8-bar label-horizon purge, equal method coverage, prediction/test-fold alignment, experiment-result row counts, fold-local artifact coverage, Phase 14A robustness artifact completeness, Phase 14B stress-grid completeness, Phase 15A/15B statistical artifact completeness, Phase 16 regime-quality artifact completeness, Phase 17 compute-plan artifact completeness, Phase 19B guided-encoder full-run artifact completeness, Phase 19A literature-positioning artifact completeness, and run-registry snapshot completeness.

The audit also records one methodological warning: the legacy `regime_assignments.csv` artifact is offline/global. Paper-grade predictive regime claims should use the Phase 13 `walkforward_experiment_results.csv` artifact instead.

Phase 13 adds that stricter benchmark. In `walkforward_regimes.py`, the regime assignment layer is refit inside each fold using only training-history rows, then applied to test rows. HMM-based test assignments use an online filtering pass initialized from the training sequence and advanced through the embargo gap. The contrastive encoder remains a frozen offline representation in this phase, but GMM/HMM assignment models on top of those embeddings are fit fold-locally.

Phase 14A adds a compact robustness matrix. The fold-local benchmark is repeated across BTC-only, ETH-only, and BTC+ETH scopes, and across 4h, 8h, and 24h triple-barrier targets. This tests whether the main result is stable across assets and prediction horizons instead of being a single-configuration artifact.

Phase 14B adds stress robustness on top of the fold-local prediction file. It does not retrain the models; it re-scores the same out-of-sample predictions under different signal thresholds, transaction costs, and market-period slices. Bull, sideways, and bear periods are defined from rolling 30-day feature-store returns. This tests whether the benchmark conclusion survives realistic deployment assumptions.

Phase 15.0 adds artifact versioning. Latest files in `models/` are still kept for the dashboard, but frozen baselines are copied into timestamped `runs/` directories with a manifest, SHA-256 hashes, git source ref, and run-index entry. The current frozen baseline is `runs/20260522_phase14b_baseline/`.

Phase 15A adds statistical rigor. It computes fold-level IC, Sharpe, total return, drawdown, and turnover from the fold-local out-of-sample prediction file, then reports bootstrap confidence intervals and paired method tests. It also runs a DM-style Newey-West forecast-loss check on multiclass negative log-likelihood, which is a calibration-oriented complement to the IC and Sharpe tests.

Phase 15B adds multiple-testing discipline. It applies Benjamini-Hochberg and Holm corrections across tested method comparisons, labels each finding as corrected, suggestive, or not significant, and adds Probabilistic Sharpe Ratio diagnostics. This prevents a single attractive p-value from becoming an overclaimed research result.

Phase 16 adds structural regime-quality metrics independent of alpha returns. It measures regime balance, persistence, posterior confidence, pairwise NMI/ARI agreement, and agreement with the raw-feature HMM reference. The HMM sequence is a classical comparison proxy, not ground truth. This phase answers whether a method produces coherent state partitions before asking whether those states improve alpha models.

Phase 17 adds compute planning before heavier encoder experiments. It profiles a synthetic encoder forward/backward step on the local machine, estimates full 30-epoch retraining cost, and creates a capped ablation queue. On the current CPU-only environment, one encoder retrain is estimated at about 99.45 minutes, and the full 12-run initial ablation grid is estimated at about 21.49 hours including evaluation overhead. The first three runs are marked as the priority queue before expanding the full grid.

Phase 18 adds the first encoder-objective upgrade. Instead of treating adjacent windows as positives by default, `guided_encoder.py` uses raw-feature HMM states as weak supervision: distant windows in the same HMM state become positives, and different-state windows near each other in the same symbol become harder negatives. The script writes separate guided artifacts and does not overwrite the existing `encoder.pt`, `regime_posteriors.csv`, or canonical benchmark files.

Phase 19A adds literature positioning. The project is now explicitly mapped against contrastive time-series representation learning, financial regime-switching models, financial ML validation, and regime-conditioned alpha modeling. This makes the paper contribution precise: the benchmark studies where learned regimes help, where classical HMM discipline remains stronger, and whether HMM-guided weak supervision can close that gap.

Dense contrastive regime inference uses stride 1 after the encoder window warmup, so the learned-regime method is compared on the same BTC+ETH row universe as HMM-style, KMeans, and volatility-bucket baselines.

## Latest Benchmark Snapshot

Latest run: BTCUSDT + ETHUSDT, `tb_label_8h`, 25,920 out-of-sample rows per method.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0165 | -0.902 | -0.909 | learned-regime benchmark |
| regime_lgbm_contrastive_hmm | 0.0035 | -0.382 | -0.852 | learned embeddings with HMM state dynamics |
| regime_lgbm_hmm | 0.0051 | -0.229 | -0.825 | best IC and Sharpe; true hmmlearn HMM |
| regime_lgbm_kmeans | -0.0013 | -1.180 | -0.920 | classical clustering baseline |
| regime_lgbm_vol_bucket | -0.0030 | -0.988 | -0.896 | simple volatility baseline |

The current result is intentionally presented as research evidence, not a profitable trading claim. The true HMM baseline is strongest in this run, while the contrastive-HMM hybrid shows that adding temporal state dynamics substantially improves the learned-regime approach.

The main interpretation is that dense contrastive inference fixed the earlier coverage problem, but unconstrained GMM clustering over high-density embeddings appears misaligned with the alpha target. HMM smoothing on embeddings repairs much of that weakness, but raw-feature HMM still produces the best downstream IC and Sharpe. This suggests useful temporal state structure matters more than embedding capacity or persistence alone.

## Fold-Local Regime Benchmark

Phase 13 reruns the regime-aware benchmark with regime assignment models refit inside each fold. This is stricter than the offline/global regime benchmark above.

| Method | IC | Sharpe | Drawdown | Note |
|---|---:|---:|---:|---|
| global_lgbm | 0.0024 | -0.506 | -0.688 | no-regime baseline |
| regime_lgbm_contrastive | -0.0110 | -0.834 | -0.926 | fold-local GMM on frozen embeddings |
| regime_lgbm_contrastive_hmm | -0.0026 | -0.548 | -0.778 | fold-local HMM on frozen embeddings |
| regime_lgbm_hmm | 0.0051 | -0.340 | -0.710 | best fold-local Sharpe |
| regime_lgbm_kmeans | 0.0072 | -0.728 | -0.860 | best fold-local IC |
| regime_lgbm_vol_bucket | -0.0020 | -0.820 | -0.854 | volatility threshold baseline |

The stricter benchmark changes the story in a useful way: fold-local KMeans has the strongest IC, while raw-feature HMM still has the strongest Sharpe and drawdown profile among regime methods. The learned contrastive-HMM path remains better than contrastive-GMM but is weaker under strict fold-local assignment than it looked in the offline/global regime benchmark.

## Phase 14A Robustness Matrix

Phase 14A repeats the strict fold-local benchmark across 9 grid cells: 3 symbol scopes by 3 target horizons.

| Scope | Target | Best IC Method | Best IC | Best Sharpe Method | Best Sharpe | Lowest Drawdown Method |
|---|---|---|---:|---|---:|---|
| BTCUSDT | tb_label_4h | regime_lgbm_hmm | 0.0016 | regime_lgbm_hmm | -0.993 | regime_lgbm_contrastive |
| BTCUSDT | tb_label_8h | regime_lgbm_kmeans | -0.0034 | regime_lgbm_contrastive | -0.546 | regime_lgbm_contrastive |
| BTCUSDT | tb_label_24h | regime_lgbm_kmeans | 0.0175 | regime_lgbm_contrastive_hmm | -0.211 | regime_lgbm_kmeans |
| ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0208 | regime_lgbm_vol_bucket | 0.315 | regime_lgbm_vol_bucket |
| ETHUSDT | tb_label_8h | global_lgbm | 0.0095 | regime_lgbm_contrastive | -0.201 | regime_lgbm_hmm |
| ETHUSDT | tb_label_24h | global_lgbm | 0.0348 | global_lgbm | 0.354 | regime_lgbm_vol_bucket |
| BTCUSDT+ETHUSDT | tb_label_4h | regime_lgbm_vol_bucket | 0.0103 | regime_lgbm_hmm | -0.205 | regime_lgbm_hmm |
| BTCUSDT+ETHUSDT | tb_label_8h | regime_lgbm_kmeans | 0.0072 | regime_lgbm_hmm | -0.340 | global_lgbm |
| BTCUSDT+ETHUSDT | tb_label_24h | regime_lgbm_contrastive_hmm | 0.0311 | regime_lgbm_vol_bucket | 0.321 | regime_lgbm_vol_bucket |

Across the grid, KMeans wins IC most often, HMM wins Sharpe most often, and volatility buckets most often win drawdown. This is stronger research evidence than a single headline result: it shows regime awareness can help, but the best regime method depends on the asset, horizon, and metric being optimized.

## Phase 14B Stress Robustness

Phase 14B reuses the strict fold-local `tb_label_8h` prediction file and stresses the trading layer across:

| Dimension | Values |
|---|---|
| Signal threshold | 0.03, 0.05, 0.07, 0.10 |
| Transaction cost | 5 bps, 10 bps, 20 bps |
| Market period | all, bull, sideways, bear |

That creates 48 stress cells and 288 method/cell rows. The goal is not to find a new best backtest; it is to see whether the existing conclusion breaks when practical assumptions change.

| Metric | Most Frequent Winner | Wins |
|---|---|---:|
| Signal IC | regime_lgbm_hmm | 24 |
| Sharpe | regime_lgbm_hmm | 22 |
| Drawdown | global_lgbm | 24 |
| Total return | regime_lgbm_hmm | 18 |

The stress test strengthens the interpretation around HMM regimes: raw-feature HMM is the most robust winner for signal IC, Sharpe, and total return across practical cost/threshold/market-period settings. The global model is the most defensive drawdown winner, especially when transaction costs rise. Contrastive-HMM remains useful in sideways regimes, but it is not yet the dominant learned-regime method.

## Phase 15A/15B Statistical Tests

Phase 15A asks whether the observed method differences are statistically reliable. The test is intentionally conservative: it uses the 18 walk-forward folds as the primary unit for IC and Sharpe tests, then uses a row-level DM-style negative-log-likelihood test as a separate calibration check.

| Method | Mean Fold IC | 95% CI Low | 95% CI High | Positive IC Folds | Mean Fold Sharpe |
|---|---:|---:|---:|---:|---:|
| regime_lgbm_hmm | 0.0058 | -0.0135 | 0.0247 | 11 | -0.561 |
| regime_lgbm_kmeans | 0.0035 | -0.0205 | 0.0276 | 8 | -0.720 |
| regime_lgbm_vol_bucket | 0.0004 | -0.0223 | 0.0234 | 10 | -0.818 |
| global_lgbm | -0.0005 | -0.0207 | 0.0209 | 9 | -0.583 |
| regime_lgbm_contrastive_hmm | -0.0063 | -0.0305 | 0.0196 | 7 | -0.908 |
| regime_lgbm_contrastive | -0.0147 | -0.0373 | 0.0095 | 7 | -0.990 |

The statistical read is more cautious than the point-estimate read. HMM has the strongest mean fold IC, but most IC and Sharpe differences versus the global model are not significant at the 5% level. Before correction, plain contrastive-GMM is worse than raw-feature HMM on fold-level IC (`p = 0.035`). After Phase 15B multiple-testing correction, this becomes a suggestive result rather than a hard claim (`BH q = 0.347` within the IC family). This supports the current research direction without overclaiming: the next improvement should not simply add a larger encoder; it should add better temporal/state constraints to the learned representation.

## Phase 16 Regime Quality

Phase 16 asks whether regime methods form coherent state partitions before using them for alpha modeling. This is deliberately separate from Sharpe or return.

| Method | Balance Entropy | Switches / 1k Bars | Transition Diagonal | Avg Duration | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| contrastive | 0.999 | 32.72 | 0.967 | 30.51 | 0.032 | 0.379 |
| contrastive_hmm | 0.999 | 23.65 | 0.976 | 42.18 | 0.020 | 0.377 |
| hmm | 0.959 | 143.24 | 0.857 | 6.98 | 1.000 | 1.000 |
| kmeans | 0.866 | 246.60 | 0.753 | 4.05 | 0.182 | 0.459 |
| vol_bucket | 1.000 | 95.71 | 0.904 | 10.44 | 0.333 | 0.599 |

The useful finding is diagnostic rather than promotional. Contrastive and contrastive-HMM produce very balanced and persistent regimes, but they weakly agree with the raw-feature HMM reference. Volatility buckets and KMeans agree more with HMM states, even though they are simpler. This supports the next research step: improve the encoder objective so learned embeddings become alpha-relevant, not merely smooth.

## Phase 17 Compute Plan

Phase 17 prevents scope creep before the encoder-upgrade phases. The project now records the current encoder cost and the next ablation queue as artifacts.

| Metric | Value | Meaning |
|---|---:|---|
| Training windows | 34,798 | Sliding windows across BTCUSDT and ETHUSDT |
| Batches per epoch | 271 | Batch size 128, drop-last |
| Encoder parameters | 139,408 | Current `TemporalEncoder` size |
| Measured step time | 0.734 sec | Synthetic CPU forward/backward step |
| Estimated 30-epoch retrain | 99.45 min | One encoder experiment |
| Initial 12-run grid | 21.49 hours | 3 losses x 2 augmentations x 2 assignment methods |
| Budget status | green | Under the 24-hour local budget |

The first three runs are:

| Priority | Loss | Augmentation | Assignment | Decision |
|---:|---|---|---|---|
| 1 | hmm_guided | time_only | hmm | run_first |
| 2 | hmm_guided | time_only | gmm | run_first |
| 3 | hmm_guided | time_frequency | hmm | run_first |

This makes Phase 18 practical: start with HMM-guided objectives and only expand to the full ablation grid if the early runs improve the learned-regime path.

## Phase 18 HMM-Guided Encoder

Phase 18 implements the first representation-learning upgrade. It uses the existing raw-feature HMM sequence as weak supervision, not ground truth. The goal is to test whether the learned embedding space becomes more state-aligned when positives and hard negatives are chosen from regime structure rather than only adjacent windows.

The one-epoch smoke run produced:

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| hmm_guided_gmm | 1 | 0.341 | 15.17 | 0.934 | 0.387 | 0.652 |
| hmm_guided_hmm | 1 | 0.353 | 18.55 | 0.946 | 0.389 | 0.620 |

This is a smoke-test result, not the final Phase 18 claim. It is still promising because the guided encoder already aligns with the HMM reference more strongly than the old contrastive encoder path from Phase 16. The full research run should use the standard 30 epochs and then feed the best guided assignments into the later statistical re-test phase.

## Phase 19A Literature Positioning

Phase 19A turns the project from a strong engineering artifact into a paper-shaped research artifact. It adds a related-work map across four areas:

- time-series contrastive learning: TS2Vec, TNC, CoST, TS-TCC, and TF-C
- financial regime switching: Hamilton-style Markov switching, Gaussian HMMs, and Markov-switching GARCH
- financial ML validation: triple-barrier labeling, purging, embargoing, and backtest overfitting control
- regime-conditioned alpha modeling: global versus state-conditioned predictors under equal validation

The key paper framing is now clear: Adaptive Alpha Lab is not claiming that HMM states are true market regimes or that the strategy is profitable. It is testing whether learned regime representations can beat, match, or explain classical sequential regimes under fair financial validation. The detailed source map is stored in `reports/related_work.md` and `reports/literature_matrix.csv`.

## Phase 19B Full Guided Encoder Run

Phase 19B runs the HMM-guided encoder for the intended 30-epoch budget. The loss fell from `0.9284` in epoch 1 to `0.0949` in epoch 30, and every batch kept valid contrastive anchors.

| Method | Epochs | Silhouette | Avg Duration | Transition Diagonal | HMM NMI | HMM Purity |
|---|---:|---:|---:|---:|---:|---:|
| hmm_guided_gmm | 30 | 0.384 | 5.09 | 0.804 | 0.609 | 0.759 |
| hmm_guided_hmm | 30 | 0.629 | 5.72 | 0.825 | 0.869 | 0.957 |

The important structural result is that `hmm_guided_hmm` is now much closer to the raw-feature HMM reference than the old learned-regime path. Phase 16 contrastive-GMM had `HMM NMI = 0.032`; the full HMM-guided HMM reaches `0.869`. This does not yet prove better alpha performance, but it does prove that the guided objective can make the learned embedding path strongly state-aligned. The downstream test comes next.

## Current Status

The codebase now produces offline/global and fold-local regime benchmarks, a validation audit, Phase 14A symbol/horizon robustness, Phase 14B cost/threshold/period stress robustness, a frozen baseline run registry, Phase 15A/15B statistical significance and multiple-testing artifacts, Phase 16 structural regime-quality diagnostics, Phase 17 compute-planning artifacts, Phase 18/19B HMM-guided encoder diagnostics, Phase 19A literature-positioning artifacts, and a Streamlit research dashboard. The next important work is not live trading; it is feeding the full guided-regime assignments into the fold-local alpha benchmark and testing whether better structural alignment improves downstream IC, Sharpe, drawdown, or calibration.


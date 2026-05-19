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
- Global LightGBM baseline and regime-conditioned LightGBM models.
- 5-day embargoed walk-forward validation.
- Validation audit for fold separation, target horizon leakage, coverage parity, and prediction alignment.
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
python src/validation_audit.py --symbols BTCUSDT ETHUSDT
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
| `models/per_regime_stats.csv` | Volatility, return, liquidity, and IC diagnostics by regime |
| `models/experiment_results.csv` | Master model comparison table |
| `models/alpha_oos_predictions.csv` | Out-of-sample model predictions |
| `models/validation_audit.csv` | Leakage, embargo, fold, coverage, and prediction-alignment audit |
| `models/fold_audit.csv` | Fold-level train/test boundary and embargo checks |
| `models/regime_stability.png` | Stability and transition-period IC dashboard panel |
| `models/phase4_dashboard.png` | Static research backtest dashboard |
| `reports/adaptive_alpha_lab_report.md` | Research note |

## Project Structure

```text
adaptive-alpha-lab/
├── dashboard.py
├── streamlit_app.py
├── reports/
│   └── adaptive_alpha_lab_report.md
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
│   ├── validation_audit.py
│   ├── alpha_models.py
│   ├── backtest.py
│   └── check.py
└── models/
    └── generated research artifacts
```

## Methodology Notes

The primary target is `tb_label_8h`, an 8-hour triple-barrier label with classes `-1`, `0`, and `+1`. The alpha score is `P(+1) - P(-1)`. A prediction becomes a trade only when the neutral class is not dominant and the absolute score exceeds the threshold.

The validation scheme uses expanding walk-forward folds with a 5-day embargo gap between train and test windows. This reduces leakage from overlapping financial labels and makes the model comparison more defensible.

Phase 12 adds a validation audit. The current audit passes all critical checks: required tables, feature/target schema, finite joined rows, 24-row target horizon loss, 18 walk-forward folds, 120-bar embargo, 8-bar label-horizon purge, equal method coverage, prediction/test-fold alignment, and experiment-result row counts.

The audit also records one methodological warning: regime assignments are currently offline/global artifacts. Alpha model train/test folds are embargoed, but paper-grade predictive regime claims require Phase 13 fold-local regime refitting.

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

## Current Status

The codebase now produces the full benchmark artifact set, a validation audit, and a Streamlit research dashboard. The next important work is not live trading; it is fold-local regime refitting, robustness studies, and testing whether better learned embeddings can close the remaining gap against the raw-feature HMM.


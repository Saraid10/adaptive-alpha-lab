# Phase 39 Fully Fold-Local Encoder Results

## Run Status

This is a **smoke validation only; no performance claim** over 1 fold(s) and 20 symbol(s). All outcomes remain development-observed and cannot be used as an untouched final test.

## Validity Contract

- Scalers, weak-supervision HMMs, contrastive pairs, encoder weights, assignment layers, and alpha models are fit inside each outer fold.
- Epochs are selected on an inner chronological validation block.
- The selected epoch count is refit on the full authorized outer-training interval.
- Outer-test metrics do not influence training or model selection.
- Vanilla and guided learned methods share the same outer folds and test coverage as classical and global baselines.

## Method Results

| method | IC | Sharpe | drawdown | total_return | turnover | n_test_rows |
| --- | --- | --- | --- | --- | --- | --- |
| global_lgbm | 0.0717 | 1.9461 | -0.1217 | 0.7933 | 0.0267 | 14400 |
| regime_lgbm_contrastive | 0.0660 | -0.1864 | -0.4465 | -0.1170 | 0.0385 | 14400 |
| regime_lgbm_contrastive_hmm | 0.0316 | 0.1748 | -0.5563 | 0.0209 | 0.0591 | 14400 |
| regime_lgbm_hmm | 0.0604 | 1.7015 | -0.2393 | 0.7570 | 0.0391 | 14400 |
| regime_lgbm_hmm_guided_gmm | 0.0474 | 1.0126 | -0.1892 | 0.3751 | 0.0406 | 14400 |
| regime_lgbm_hmm_guided_hmm | 0.0503 | 1.2306 | -0.1556 | 0.5401 | 0.0365 | 14400 |
| regime_lgbm_kmeans | 0.0817 | 1.9767 | -0.1191 | 1.0324 | 0.0388 | 14400 |
| regime_lgbm_vol_bucket | 0.0597 | 0.9394 | -0.2240 | 0.3630 | 0.0425 | 14400 |

## Selected Epochs

| encoder_method | min | median | max |
| --- | --- | --- | --- |
| guided | 1 | 1.0000 | 1 |
| vanilla | 1 | 1.0000 | 1 |

## Interpretation Rule

A smoke run validates code paths, leakage boundaries, artifacts, and coverage only. A full run is still development evidence. Model changes require a new registered candidate family, and confirmatory claims require a frozen configuration evaluated once on a locked holdout.

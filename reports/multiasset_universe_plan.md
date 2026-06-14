# Multi-Asset Universe Plan

## Purpose

Phase 31 upgrades Adaptive Alpha Lab from a BTC/ETH pilot into a reproducible multi-asset research program. The goal is not to add symbols for appearance. The goal is to define a pre-registered crypto universe protocol before running broad experiments, so the final paper can make generalization claims without looking cherry-picked.

## Research Role

The BTC/ETH benchmark remains the controlled pilot. It isolates the core mechanism: sequential assignment makes learned regimes more useful, and HMM-guided weak supervision improves structural regime alignment.

The multi-asset stage tests whether that mechanism survives outside the pilot setting. The final paper should treat BTC/ETH as Phase 1 evidence and Crypto-20/Crypto-50 as the generalization study.

## Universe Construction

The candidate list starts from liquid Binance USDT spot pairs and excludes stablecoin pairs, leveraged tokens, and synthetic instruments. Final eligibility is determined only after ingestion and quality checks.

Selection rules:

- Use hourly Binance spot OHLCV data.
- Require continuous history over the study window whenever possible.
- Prefer symbols with at least the configured minimum bar count.
- Reject symbols with excessive missing-bar gaps.
- Rank eligible assets by median quote volume.
- Preserve a full exclusion log so failed assets are visible.

## Two-Stage Expansion

1. Crypto-20 pilot: first scalability checkpoint for data, features, targets, regimes, alpha models, and dashboard artifacts.
2. Crypto-50 final: publication-level generalization experiment after the Crypto-20 pipeline is stable.

## Reviewer-Safe Claim

The paper should not claim universal market generalization after Crypto-50. It can claim broad crypto-universe evidence under a controlled exchange, interval, and feature protocol.

Preferred wording:

> We first establish the mechanism in a controlled BTC/ETH pilot, then test whether the same regime-learning conclusions survive across a pre-specified liquid Binance USDT crypto universe.

## Required Artifacts

- `configs/crypto_universe_candidates.csv`
- `models/asset_universe_crypto20.csv`
- `models/asset_universe_crypto50.csv`
- `models/asset_universe_exclusions.csv`
- `models/asset_universe_summary.csv`

## Decision Gate

Proceed from Crypto-20 to Crypto-50 only if:

- ingestion and feature generation complete without major gaps,
- every selected symbol has enough target rows for walk-forward folds,
- validation audit passes,
- runtime and memory remain manageable,
- no single symbol family dominates the selected universe.

If Crypto-20 fails, the paper should remain a BTC/ETH controlled benchmark and describe multi-asset generalization as future work.

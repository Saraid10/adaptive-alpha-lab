# Adaptive Alpha Lab Hypotheses

## Purpose

This file freezes the paper hypotheses after Phase 23. Every future experiment should map to one or more rows in this table.

## Hypothesis Table

| ID | Hypothesis | Current Evidence | Current Status | Next Test |
|---|---|---|---|---|
| H1 | Sequential consistency matters for useful regime assignment. | Raw-feature HMM beats vanilla contrastive-GMM on the original benchmark; contrastive-HMM improves over contrastive-GMM; guided embeddings need HMM assignment to become useful; Phase 25/26 support the assignment-layer mechanism. | Supported | Include as the central mechanism claim in the paper draft. |
| H2 | Vanilla contrastive regimes are structurally stable but not alpha-relevant enough. | Phase 16 shows low HMM agreement for vanilla contrastive regimes; downstream IC is worse than raw-feature HMM. | Supported | Include vanilla contrastive as a fixed ablation reference. |
| H3 | HMM-guided contrastive supervision improves learned regime structure. | Phase 19B guided-HMM reaches `HMM NMI = 0.869` and `HMM purity = 0.957`, far above vanilla contrastive alignment; Phase 25 keeps objective guidance as supported for the guided-HMM downstream path but mixed structurally. | Strongly supported structurally | Include as structural evidence; avoid calling HMM states ground truth. |
| H4 | Better guided regime structure improves downstream alpha. | Phase 20 guided-HMM has the best primary point estimates and positive Sharpe; Phase 26 keeps the edge over raw HMM directional but statistically inconclusive. | Directionally supported, statistically inconclusive | Treat as promising evidence, not a dominance claim. |
| H5 | Guided-HMM robustness survives realistic trading assumptions. | Phase 21 stress grid shows guided-HMM as the most frequent winner across IC, Sharpe, drawdown, and total return under threshold/cost/period variation. | Supported on primary BTC+ETH 8h setup | Keep stress result as robustness evidence; do not overgeneralize. |
| H6 | Guided-HMM regimes are economically interpretable. | Phase 23 shows fold-local feature attribution dominated by volatility state, momentum/autocorrelation, and distribution-shape features. | Supported diagnostically | Include interpretability in paper; avoid causal language. |
| H7 | The structural regime-learning mechanism transfers beyond BTC/ETH. | Phase 31-35 support structural transfer. Phase 36/37 give guided-HMM the highest mean fold IC, but its edge over raw HMM is non-significant (`p=0.840`) and its probability calibration is worse than global LightGBM. | Structurally supported, predictive dominance not supported | Discuss structural transfer separately from portfolio-level and calibration claims. |
| H8 | The structural and assignment-layer findings survive fully fold-local encoder training. | Phase 39 now implements and leakage-tests fold-local scaling, weak-supervision HMMs, pair mining, encoders, assignments, and alpha models. The one-fold run is a smoke test only. | Implementation validated; scientific result open | Execute the frozen full development protocol without changing it in response to smoke metrics. |
| H9 | Validation-only calibration and soft posterior gating improve probability quality without sacrificing ranking IC. | Phase 37 identifies worse guided-HMM NLL despite weak positive IC direction. | Open | Test only after the fold-local baseline passes; select using inner validation. |
| H10 | The selected conclusion transfers to an untouched asset or temporal holdout. | Existing BTC/ETH and Crypto-20 outcomes have been inspected and are development-observed. | Open | Freeze one candidate and run a single locked external evaluation. |

## Phase 25 Status Update

The minimal ablation suite supports the assignment-layer mechanism most strongly: HMM assignment improves the guided learned-regime path more reliably than GMM assignment. Objective guidance is useful for the guided-HMM downstream path, while the current time-frequency prototype remains a negative or inconclusive ablation.

## Phase 26 Status Update

The paper statistical refresh keeps the mechanism claim strong and the alpha dominance claim cautious. HMM assignment on guided embeddings is raw-suggestive on fold-level IC versus guided-GMM (`p=0.075`) and improves all focused point-estimate metrics. Guided-HMM versus raw-feature HMM remains directionally supported, not statistically significant.

## Primary Paper Hypothesis

The primary paper hypothesis is H4:

```text
HMM-guided contrastive regime representations, when paired with sequential HMM assignment, improve regime-conditioned alpha modeling versus raw-feature HMM and vanilla learned-regime baselines under purged walk-forward validation.
```

Current status:

```text
Directionally supported by point estimates and stress robustness, not yet statistically conclusive at fold level.
```

## Secondary Hypotheses

H1, H2, H3, and H6 are already strong enough to support the paper narrative.

H5 supports robustness on the primary setup.

H7 is now part of the paper's multi-asset mechanism evidence, with predictive dominance explicitly unsupported.

## Phase 38 Hypothesis Hierarchy

H8 is the next validity hypothesis and must be resolved before a new model-improvement claim. H9 is the bounded development hypothesis motivated by the Phase 37 calibration failure. H10 is the confirmatory generalization hypothesis and cannot be tested until one configuration is frozen.

The historical H4 result remains directionally supported in the BTC/ETH pilot and unsupported as a broad Crypto-20 dominance claim. A failed H8, H9, or H10 test must remain in the final evidence record and must not trigger tuning on the same outer or locked data.

## Phase 39 Status Update

The Phase 39 implementation gate passes on seven boundary/reproducibility tests and one end-to-end fold with equal coverage across all eight methods. This validates the experimental machinery, not H8. The smoke metrics are non-evidentiary; H8 remains open until the pre-specified full development run is completed.

## Claim Language

Use:

```text
The guided-HMM method produced the strongest point estimates and stress robustness in the BTC/ETH benchmark, while fold-level statistical dominance over raw-feature HMM remains inconclusive.
```

Avoid:

```text
The guided-HMM method beats HMM.
The learned regimes are true market regimes.
The strategy is profitable.
The method generalizes to all markets.
```


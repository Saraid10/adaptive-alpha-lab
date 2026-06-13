# Adaptive Alpha Lab Hypotheses

## Purpose

This file freezes the paper hypotheses after Phase 23. Every future experiment should map to one or more rows in this table.

## Hypothesis Table

| ID | Hypothesis | Current Evidence | Current Status | Next Test |
|---|---|---|---|---|
| H1 | Sequential consistency matters for useful regime assignment. | Raw-feature HMM beats vanilla contrastive-GMM on the original benchmark; contrastive-HMM improves over contrastive-GMM; guided embeddings need HMM assignment to become useful; Phase 25 supports the assignment-layer mechanism. | Supported | Refresh statistical evidence in Phase 26. |
| H2 | Vanilla contrastive regimes are structurally stable but not alpha-relevant enough. | Phase 16 shows low HMM agreement for vanilla contrastive regimes; downstream IC is worse than raw-feature HMM. | Supported | Include vanilla contrastive as a fixed ablation reference. |
| H3 | HMM-guided contrastive supervision improves learned regime structure. | Phase 19B guided-HMM reaches `HMM NMI = 0.869` and `HMM purity = 0.957`, far above vanilla contrastive alignment; Phase 25 keeps objective guidance as supported for the guided-HMM downstream path but mixed structurally. | Strongly supported structurally | Keep claim directional until Phase 26 refreshes statistical evidence. |
| H4 | Better guided regime structure improves downstream alpha. | Phase 20 guided-HMM has the best primary point estimates and positive Sharpe, but fold-level IC edge over raw HMM is not significant at 5%. | Directionally supported, statistically inconclusive | Refresh statistical tests after Phase 25 ablations. |
| H5 | Guided-HMM robustness survives realistic trading assumptions. | Phase 21 stress grid shows guided-HMM as the most frequent winner across IC, Sharpe, drawdown, and total return under threshold/cost/period variation. | Supported on primary BTC+ETH 8h setup | Keep stress result as robustness evidence; do not overgeneralize. |
| H6 | Guided-HMM regimes are economically interpretable. | Phase 23 shows fold-local feature attribution dominated by volatility state, momentum/autocorrelation, and distribution-shape features. | Supported diagnostically | Include interpretability in paper; avoid causal language. |
| H7 | The result generalizes across asset classes. | Not tested. | Open | Conditional multi-asset gate after Phase 25/26. |

## Phase 25 Status Update

The minimal ablation suite supports the assignment-layer mechanism most strongly: HMM assignment improves the guided learned-regime path more reliably than GMM assignment. Objective guidance is useful for the guided-HMM downstream path, while the current time-frequency prototype remains a negative or inconclusive ablation. These findings narrow Phase 26 to statistical evidence refresh rather than more model expansion.

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

H7 is not required for the first paper draft.

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


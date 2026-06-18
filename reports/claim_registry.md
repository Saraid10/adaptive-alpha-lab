# Adaptive Alpha Lab Claim Registry

## Purpose

This registry separates allowed paper claims from claims that are not supported by the current evidence. It should be consulted before editing the README, report, dashboard, paper draft, or resume bullets.

## Claim Status Levels

| Status | Meaning |
|---|---|
| Supported | Evidence is strong enough for the main text with normal caveats. |
| Directional | Point estimates or diagnostics support the idea, but statistical evidence is incomplete. |
| Diagnostic | Useful for explanation or mechanism, not a causal or predictive claim. |
| Open | Not tested yet. |
| Forbidden | Should not be claimed. |

## Supported Claims

| Claim | Status | Evidence Artifact | Safe Wording |
|---|---|---|---|
| Vanilla contrastive-GMM regimes are weak in this benchmark. | Supported | `models/walkforward_experiment_results.csv`, `models/regime_quality_summary.csv`, `models/statistical_claims.csv` | Vanilla contrastive regimes underperform raw-feature HMM and show weak alignment with the HMM reference. |
| Sequential assignment matters. | Supported | `models/guided_alpha_comparison.csv`, `models/guided_encoder_comparison.csv`, `models/paper_statistical_summary.csv` | Learned embeddings are more useful when the assignment layer preserves temporal state dynamics. |
| HMM-guided supervision improves structural regime alignment. | Supported | `models/guided_encoder_summary.csv`, `models/guided_encoder_comparison.csv` | HMM-guided weak supervision substantially increases agreement with the raw-feature HMM reference. |
| Guided-HMM is stress-robust on the primary BTC+ETH 8h prediction file. | Supported | `models/robustness_stress_summary.csv`, `models/robustness_stress_wins.csv` | Guided-HMM is the most frequent winner in the refreshed primary stress grid. |
| Guided-HMM feature attribution is economically plausible. | Diagnostic | `models/feature_importance_by_regime.csv`, `models/feature_family_summary.csv` | Fold-local interpretability shows the alpha model relies mainly on volatility state, momentum/autocorrelation, and distribution shape. |
| HMM assignment is the strongest current ablation mechanism. | Supported | `models/ablation_summary.csv`, `models/paper_statistical_summary.csv`, `models/ablation_heatmap.png` | Phase 25/26 show the assignment layer is the most consistently supported mechanism across structural and downstream comparisons. |

## Directional Claims

| Claim | Status | Evidence Artifact | Safe Wording |
|---|---|---|---|
| Guided-HMM improves downstream alpha over raw-feature HMM. | Directional | `models/walkforward_experiment_results.csv`, `models/ablation_summary.csv`, `models/paper_statistical_summary.csv`, `models/statistical_pairwise_tests.csv`, `models/statistical_claims.csv` | Guided-HMM improves point estimates over raw-feature HMM on the primary benchmark, but Phase 26 confirms the fold-level IC evidence remains statistically inconclusive. |
| Guided-HMM has better risk-adjusted behavior than raw-feature HMM. | Directional | `models/statistical_sharpe_diagnostics.csv`, `models/robustness_stress_summary.csv` | Guided-HMM has the strongest PSR diagnostic and stress-grid performance, but this is not a deployable performance claim. |

## Diagnostic Claims

| Claim | Status | Evidence Artifact | Safe Wording |
|---|---|---|---|
| Structural regime-learning behavior transfers beyond BTC/ETH. | Diagnostic | `models/crypto20_guided_encoder_summary.csv`, `models/crypto20_guided_encoder_comparison.csv`, `reports/multiasset_universe_plan.md` | Phase 35 shows the HMM-guided objective scales structurally to Crypto-20, but this is not evidence of downstream alpha generalization. |

## Open Claims

| Claim | Status | Required Evidence |
|---|---|---|
| Time-frequency guided encoding improves the full model. | Open | Full-length time-frequency run plus downstream alpha retest; Phase 25 says not to expand this yet. |
| Hard-negative mining improves guided regimes. | Open | Capped ablation suite. |
| Downstream alpha results generalize beyond BTC/ETH. | Directional only | Phase 37 gives guided-HMM the highest mean fold IC, but the edge is not significant after paired testing or correction; Sharpe/return dominance is unsupported. |
| Fold-local encoder retraining changes the conclusion. | Open | Expanding-window or fold-local encoder retraining experiment. |

## Forbidden Claims

| Claim | Why Forbidden |
|---|---|
| HMM states are true market regimes. | HMM states are proxy/reference states, not ground-truth labels. |
| The trading strategy is profitable. | Backtest returns are research diagnostics with limited assets and overlapping labels. |
| Guided-HMM statistically dominates raw-feature HMM. | The current fold-level IC edge is not significant at 5%. |
| The result generalizes to equities, FX, or commodities. | The expanded tests are still crypto-only and do not cover other asset classes. |
| The result statistically improves downstream alpha on Crypto-20. | Unsupported. Phase 37 IC confidence intervals cross zero and corrected tests do not establish superiority over global, raw HMM, or KMeans. |
| The guided model is better calibrated on Crypto-20. | Unsupported. Time-block DM tests show worse multiclass NLL than global LightGBM and raw-feature HMM. |
| SHAP proves causal market drivers. | Feature attribution is model-specific and diagnostic. |
| Offline/global regime assignments prove predictive performance. | Predictive claims require fold-local regime refits. |

## Resume-Safe Language

Use:

```text
Built a research-grade quant ML benchmark comparing classical and learned market-regime methods under financial labels, purged walk-forward validation, transaction costs, statistical tests, and fold-local interpretability.
```

Use:

```text
Found that HMM-guided contrastive regimes produced the strongest BTC/ETH point estimates and stress robustness, while statistical dominance over raw-feature HMM remained inconclusive.
```

Avoid:

```text
Built a profitable trading bot.
Discovered true market regimes.
Created a novel SOTA alpha model.
```


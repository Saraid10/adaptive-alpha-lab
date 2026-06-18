# Paper Submission Checklist

## Phase 38 Status

This checklist tracks what remains after the completed Phase 37 Crypto-20 adjudication. Passing the historical audit does not by itself make the project submission-ready.

## Ready

- Central research question is frozen in `reports/paper_protocol.md`.
- Claim boundaries are frozen in `reports/claim_registry.md`.
- Literature positioning exists in `reports/related_work.md` and `reports/literature_matrix.csv`.
- Main fold-local benchmark artifacts exist.
- Phase 25 ablations and Phase 26 paper claim tests exist.
- Phase 29 manuscript prose pass and Phase 37 evidence synchronization are complete.
- Validation audit has no critical failures.
- Crypto-20 structural, predictive, calibration, and portfolio outcomes are separated honestly.
- Phase 38 data roles, experiment ledger, and publication gates are defined.

## Critical Scientific Work

- Retrain the learned encoder fully inside each outer fold.
- Add the missing vanilla contrastive and contrastive-HMM Crypto-20 baselines.
- Use inner chronological validation for epochs, calibration, thresholds, and model selection.
- Freeze exactly one candidate before an untouched external evaluation.
- Add dependence-aware time-and-asset inference for the final claim table.
- Harden execution timing, costs, overlapping positions, and exposure accounting.

## Critical Paper Work

- Refresh related work through the submission year.
- Replace planning prose with anonymous ACM `sigconf` LaTeX.
- Fit the complete paper, including figures and references, inside the venue limit.
- Make the paper self-contained without relying on an appendix.
- Map every quantitative claim to a generated artifact.
- Complete anonymity, citation, formatting, and claim audits.

## Explicitly Deferred

- Crypto-50 expansion before fold-local validity is established.
- Unrestricted architecture or feature search.
- Full time-frequency training without a reopened written gate.
- FastAPI, paper trading, and dashboard expansion before model freeze.
- Any real-money execution.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim statistically proven Crypto-20 alpha, calibration, or portfolio-performance dominance.
- Do not treat Phase 35 structural Crypto-20 results as trading or alpha evidence.
- Do not describe the inspected Phase 36/37 Crypto-20 results as an untouched final test.
- Do not treat fold-local assignment as sufficient when the learned encoder was trained offline.

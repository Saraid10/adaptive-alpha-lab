# Paper Submission Checklist

## Phase 27 Status

This checklist tracks what remains before the project should be treated as a submission-ready paper package.

## Ready

- Central research question is frozen in `reports/paper_protocol.md`.
- Claim boundaries are frozen in `reports/claim_registry.md`.
- Literature positioning exists in `reports/related_work.md` and `reports/literature_matrix.csv`.
- Main fold-local benchmark artifacts exist.
- Phase 25 ablations and Phase 26 paper claim tests exist.
- Validation audit has no critical failures.

## Needs Human Writing

- Convert `paper/main.md` from scaffold to prose.
- Replace placeholder citations with the final venue citation style.
- Add final figure numbers after choosing the paper template.
- Tighten the abstract after the final submission venue is selected.
- Decide whether the appendix includes the Streamlit dashboard screenshots.

## Optional Before Submission

- Multi-asset generalization only if scope and compute allow it.
- Fold-local encoder retraining only if the paper needs a stronger leakage-resistance appendix.
- Full time-frequency encoder run only if Phase 25/26 gates are reopened.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim generalization outside BTC/ETH without new experiments.

# Paper Submission Checklist

## Phase 29 Status

This checklist tracks what remains before the project should be treated as a submission-ready paper package.

## Ready

- Central research question is frozen in `reports/paper_protocol.md`.
- Claim boundaries are frozen in `reports/claim_registry.md`.
- Literature positioning exists in `reports/related_work.md` and `reports/literature_matrix.csv`.
- Main fold-local benchmark artifacts exist.
- Phase 25 ablations and Phase 26 paper claim tests exist.
- Phase 29 manuscript prose pass is complete.
- Validation audit has no critical failures.

## Needs Human Writing

- Replace paper-planning source names with the final venue citation style.
- Add final figure numbers after choosing the paper template.
- Tune the abstract to the final submission venue length.
- Decide whether the appendix includes the Streamlit dashboard screenshots.

## Optional Before Submission

- Crypto-20 downstream alpha retest only if the paper needs a predictive generalization section.
- Crypto-50 expansion only if Crypto-20 downstream evidence and compute budget justify it.
- Fold-local encoder retraining only if the paper needs a stronger leakage-resistance appendix.
- Full time-frequency encoder run only if Phase 25/26 gates are reopened.

## Must Not Claim

- Do not claim HMM states are ground truth.
- Do not claim a profitable or deployable trading strategy.
- Do not claim guided-HMM statistically dominates raw-feature HMM at 5%.
- Do not claim downstream alpha generalization outside BTC/ETH before the Crypto-20 fold-local alpha retest.
- Do not treat Phase 35 structural Crypto-20 results as trading or alpha evidence.

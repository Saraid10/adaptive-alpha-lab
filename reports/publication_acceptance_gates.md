# Publication And Project Acceptance Gates

## Purpose

These gates prevent scope growth, test-set tuning, and paper claims that are stronger than the evidence. A later gate cannot pass while an earlier critical gate is open.

## Gate 0: Evidence Freeze

Pass criteria:

- Phase 37 artifacts are archived in Git history and tagged.
- Existing BTC/ETH and Crypto-20 results are marked `development_observed`.
- The experiment ledger records inspected and uninspected families.
- No historical prediction artifact is silently replaced.

Failure action: stop new modeling and repair the registry.

## Gate 1: Fold-Local Validity

Pass criteria:

- Scaling, HMM fitting, pair mining, encoder training, assignment, calibration, thresholding, and alpha fitting are bounded by the outer-training interval.
- Epoch and candidate selection use inner chronological validation only.
- Automated tests reject future-row contamination.
- Every fold records train, validation, embargo, purge, and test boundaries.

Failure action: results are diagnostic only and cannot support the final predictive claim.

## Gate 2: Baseline Completeness And Parity

Pass criteria:

- Global, classical, vanilla learned, and guided learned methods are present.
- GMM and HMM assignment isolate the sequential-assignment effect.
- Methods share target, folds, costs, and test coverage.
- Missing predictions, duplicate rows, and coverage mismatches are zero.

Failure action: do not interpret method rankings.

## Gate 3: Candidate Selection

Pass criteria:

- Candidate family was authorized before results were inspected.
- Selection uses development and inner-validation evidence only.
- Primary IC and calibration endpoints are reported together.
- Rejected candidates remain in the experiment ledger.
- Exactly one configuration is frozen for external evaluation.

Failure action: keep the Phase 37 mechanism/negative-result paper path.

## Gate 4: Locked External Evaluation

Pass criteria:

- Holdout membership and hashes are recorded before execution.
- The candidate configuration is frozen before any holdout outcome is viewed.
- Evaluation is executed once.
- Failure or contradiction is reported without same-holdout retuning.

Failure action: prohibit broad predictive-generalization and superiority claims.

## Gate 5: Statistical Adjudication

Pass criteria:

- Primary tests operate on defensible dependent-data units.
- Confidence intervals and practical effect sizes accompany p-values.
- Multiple-testing families are declared before correction.
- Time and asset dependence are acknowledged or modeled.
- Economic metrics include realistic costs and execution timing.

Failure action: downgrade the affected claim to directional or diagnostic.

## Gate 6: ICAIF Paper Readiness

Pass criteria:

- Anonymous ACM `sigconf` PDF is at most eight total pages including references.
- The paper is self-contained and does not rely on an appendix.
- Literature is current through the submission year.
- Every quantitative claim maps to a generated artifact.
- The abstract states the primary negative or inconclusive result when applicable.
- Anonymity, citation, formatting, and claim audits pass.

Failure action: do not submit a known-invalid or over-length manuscript.

## Gate 7: BTech Project Readiness

Pass criteria:

- One-command research smoke run succeeds.
- API, dashboard, and historical replay use the frozen model contract.
- No real-money execution is enabled.
- Tests, environment lock, artifact lineage, architecture documentation, demonstration video, report, and viva deck exist.
- The demo can run without private credentials or live-network dependence.

Failure action: treat the repository as a research prototype rather than a completed product.

## Current Gate State

| Gate | State after Phase 37 | Phase 38 action |
|---|---|---|
| 0 Evidence Freeze | Nearly complete | Enforce registries and immutable interpretation |
| 1 Fold-Local Validity | Open | Make fold-local encoder the next critical implementation |
| 2 Baseline Completeness | Open | Add dense vanilla Crypto-20 learned baselines |
| 3 Candidate Selection | Blocked | Wait for Gates 1 and 2 |
| 4 Locked Evaluation | Blocked | Define holdout now; execute after model freeze |
| 5 Statistical Adjudication | Development version complete | Extend only after locked evaluation |
| 6 Paper Readiness | Open | Refresh literature and produce eight-page ACM paper later |
| 7 BTech Project Readiness | Open | Productize after research model freeze |

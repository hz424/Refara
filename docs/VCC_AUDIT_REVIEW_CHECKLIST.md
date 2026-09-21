# Independent VCC audit review

Use this checklist to verify an audit against the pinned official scorer.
Call that scorer directly to calculate expected values, independently of the
adapter's numerical helpers. Save code and input hashes, upstream outputs and
adapter results together. On a cluster, run numerical checks in a scheduled job.

| Check | Evidence required |
|---|---|
| Existing MSE behavior | Allocation, scoring, effect/state representation and identity tests pass with the existing public API |
| Upstream execution | Actual imported cell-eval2 code matches the pinned commit; scorer Python, dependencies, backend, device and resolved configuration are recorded |
| Input contract | Raw-cell files pass the upstream count, label and gene-axis checks. Invalid counts and duplicate or missing identifiers are rejected |
| Membership | Recorded cell IDs recover the selected source rows and depth in each stratum. Shared assignments use identical cells; separated assignments have no cell overlap within a dataset |
| Fixed panel | All models use the same non-control reference cells, perturbations and gene axis. Missing models or tasks cannot silently narrow the comparison |
| Model inputs | Model-input and held-out scoring pools remain distinct and are recorded separately. Each input intervention links its prediction to the input pool and generation rule. The public example changes input cells while preserving the rule hash and seed |
| Prediction integrity | Original prediction-file hashes and non-control predictions remain unchanged. Every control inserted into or replaced in a scoring view is recorded |
| Official baseline parity | On the original shared panel and a declared separated panel, all model arms match a direct official call for metric aggregates, baseline values, replicate values, metric-level calibrated scores and calibrated aggregate, within a declared numerical tolerance; the prespecified shared rejection also reproduces the same narrow official unavailability outcome |
| Panel aggregation | A case with unequal perturbation expression-error denominators agrees with the official ratio of sums; discrimination includes the full competitor panel and preserves upstream ties |
| Calibration identity | Calibration matches the assigned observation controls and all scored models in the allocation. Baseline and model views use the same prediction-reference controls. Incompatible bundles and altered metadata are rejected |
| Anchor semantics | Anchor receipts retain per-half controls, pinned split seeds and metric-specific estimators, including the full-reference gate used by the relevant fold-change metric |
| Diagnostic scoring | Separated-reference arms are marked diagnostic. Their summaries use official `from_replicate` values at `avg_score`; `from_baseline` is a different quantity |
| Missing/degenerate evidence | Missing receipts and unexpected errors remain failures. A recognized baseline rejection retains every model, metric and pair as NA, with the original reason, validated inputs and completed-with-unavailable status. No reduced-metric mean replaces an unavailable score |
| Pairwise summary | Every model pair has a stated orientation. Equal scores remain ties. Summaries distinguish raw metric direction from calibrated scores, for which higher is better |
| Failure and rerun | Unexpected execution errors or changed inputs stop dependent work. Only recognized official baseline-scale failures count as calibration unavailability. Failed and unavailable outputs remain available for review; changed inputs require new completion receipts |
| Units and interpretation | Repeated allocations describe within-data sensitivity. Cell disjointness alone does not establish biological independence. Unit-level inference states its experimental units and assumptions |
| Public demonstration | The example records the source, terms, subset and prediction construction. Its results establish agreement with the official scorer on these data; held-out competition performance requires separate evaluation |

For each executed row, record `PASS`, `FAIL` or `NOT_RUN`, the comparison performed,
the relevant output paths and any numerical tolerance. List unresolved checks
in the release review.

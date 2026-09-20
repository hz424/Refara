# Independent VCC audit review

Use this checklist to review an audit. Run numerical checks in the scheduled
QA environment, save code/input hashes and keep upstream output alongside the
adapter result. Compute expected values with a direct call to the pinned
upstream scorer, independently of the adapter's numerical helpers.

| Check | Evidence required |
|---|---|
| Existing MSE behavior | Existing allocation, score, effect/state representation and identity tests pass without a changed public API |
| Upstream execution | Actual imported cell-eval2 code matches the pinned commit; scorer Python, dependencies, backend, device and resolved configuration are recorded |
| Input contract | Real raw-cell files satisfy upstream count, label and gene-axis validation; duplicate/missing labels or genes and invalid counts fail visibly |
| Membership | Output cell IDs reconstruct the selected source rows and actual per-stratum depths; shared means identical membership, separated means no intersection within the source identity namespace |
| Fixed panel | All model arms use the same non-control reference cells, perturbation support and gene axis; a missing arm or task cannot silently narrow the comparison |
| Model inputs | Public input pools remain distinct from held-out scoring pools; every conditioning intervention has a prediction linked to its named input and generation procedure; the positive public example changes actual input cells while retaining the parent generation-rule hash and seed |
| Prediction integrity | Original prediction files have unchanged hashes; assembled scoring views have exactly the original non-control payload, with all inserted/replaced controls recorded |
| Official baseline parity | On the original shared panel and a declared separated panel, all model arms match a direct official call for metric aggregates, baseline values, replicate values, metric-level calibrated scores and calibrated aggregate, within a declared numerical tolerance; the prespecified shared rejection also reproduces the same narrow official unavailability outcome |
| Panel aggregation | A case with unequal perturbation expression-error denominators agrees with the official ratio of sums; discrimination includes the full competitor panel and preserves upstream ties |
| Calibration identity | Changing real control membership invalidates an incompatible old bundle; baseline and model scoring views use the same assigned prediction-reference controls; all scored models within an allocation share the same bound calibration; stale/tampered metadata is rejected |
| Anchor semantics | Anchor receipts retain per-half controls, pinned split seeds and metric-specific estimators, including the full-reference gate used by the relevant fold-change metric |
| Diagnostic scoring | A separated-reference arm is marked diagnostic and retains official `from_replicate` scores; the summary reads `from_replicate` at `avg_score`, not the diagnostic `from_baseline` average |
| Missing/degenerate evidence | Missing receipts and unexpected errors remain failures; a recognized official baseline rejection retains all model/metric/pair slots as explicit NA with its original reason, validated input bindings and completed-with-unavailable status; no reduced-metric mean or all-scores-available status substitutes for it |
| Pairwise summary | Each declared model pair appears with explicit orientation; equal scores remain ties; raw metric direction and calibrated higher-is-better direction are not confused |
| Failure and rerun | An unexpected execution error or input change stops dependent scoring/summary; recognized baseline-scale unavailability is handled only by its narrow official exception contract; original failed and unavailable artifacts remain reviewable; a new input cannot inherit an old completion receipt |
| Units and interpretation | Repeated allocations produce descriptive comparisons only; cell disjointness is not labelled verified biological independence; any separate unit-level inference carries its own declared support and assumptions |
| Public demonstration | The example records its resource, source terms, subset and frozen prediction construction; output describes software parity on those data and makes no hidden-leaderboard claim |

For each executed row, record `PASS`, `FAIL` or `NOT_RUN`, the comparison performed,
the relevant output paths and any numerical tolerance. List unresolved checks
in the release review.

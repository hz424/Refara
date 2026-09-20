# V22 frozen attribution design

Frozen on 31 August 2026 before the V22 pilot and formal analyses. This
document and `v22_frozen_config.json` specify the design; V21 was not modified.

The wording in this public copy was edited for clarity. Numerical settings,
identifiers and decisions are unchanged from the frozen production file
(10,129 bytes; SHA-256
`7c27ca38c1328ae99412dea3f9ccf888d146ed140ec9a3f89b81c471e36b4c8a`).

**Pre-model amendment 001.** The initial Ensembl/count identity check failed
before any model or endpoint was run. The amendment files record that failure
and replace the invalid feature-identity bridge; endpoints, models,
aggregation, inference and reporting criteria were unchanged.

> A fixed eight-donor factor-isolation analysis tested whether output representation, prediction centering and model conditioning altered benchmark contrasts under a leave-one-donor-out learning pipeline. Endpoints were specified before this analysis, but their directions were informed by earlier results from the same resource.

The three factors are isolated through fixed design contrasts. This does not
assert biological, statistical or causal independence.

## Scientific scope

This was a nonblinded, endpoint-prespecified analysis of a fixed eight-donor
panel not used in V21. Aggregate results from the same dataset had previously
been examined, and the leave-one-donor-out fits share training donors. We
therefore interpret V22 as within-panel factor isolation, not prospective or
independent validation, donor-population inference, prevalence estimation or
causal analysis.

## Data, roots and factor geometry

- Dataset: Kang et al. IFN-beta PBMC (`GSE96583`; ExperimentHub `EH2259`), with
  the existing 13,093-cell KangCrossPatient object used for frozen metadata and
  identity checks.
- Roots: eight donors. Each LODO fold excludes every cell from the held-out
  donor from gene selection, preprocessing and model fitting.
- Tasks: B, CD14 Mono, CD16 Mono and CD4 T, weighted equally.
- Within every held-out donor-task, an expression-blind SHA-256 rule selects 12
  controls and 12 stimulated cells. The controls are split into three disjoint
  physical blocks of four cells. All six mappings of those blocks to semantic
  labels A, B and C are evaluated. Contrasts are formed within a mapping before
  mappings are averaged; consequently each physical block occupies every label
  exactly twice.
- A `Cmodel` block is a donor-by-task context bundle: the same four cells supply
  both the source cells and their mean context embedding in training-fold PCA
  space. It is not called a donor-wide embedding.

Before any model run, raw identity must pass exact barcode and full frozen-
metadata equality, followed by a unique deterministic 5,000-feature
symbol/count bridge. The frozen bridge comprises 4,980 literal-symbol exact-
count routes, three Seurat `make.unique` base-symbol exact-count routes and 17
unambiguous source-version-ledger routes. Those 17 routes are accepted only if
the complete 33-entry official-minus-anchor discrepancy ledger equals the
frozen ledger exactly (17 genes, 33 cells, all differences positive, sum 35,
maximum 2; SHA-256
`95e672201987a9c32c6471af29f730487e355d7ccdb35e3fec5dcfa13432fb0b`).
This is an identity equality rule, not a count tolerance or fuzzy match; every
mapped raw column must be unique. Any mismatch remains `NO_GO_RAW_IDENTITY`.
Official EH2259 full 35,635-gene raw counts are the sole input to all V22
preprocessing; the processed object supplies only frozen cell identity,
metadata and reconciliation anchors, never a 5,000-gene expression fallback.
Published cell filtering and annotations remain external frozen dataset
definitions.

## Fold-only preprocessing and models

Within each seven-donor training fold, all control and stimulated cells from the
four tasks undergo CP10K followed by natural `log1p`. Genes are ranked by sample
variance over these training rows; ties are broken by original raw-gene index.
The top 5,000 are retained and restored to original index order. A centered,
unscaled 100-component PCA is fitted only on these training rows. Gene scales
for the primary utility are sample standard deviations computed only from
training controls, floored at 0.1.

The matched baseline is a control-profile low-rank linear ridge model. For each
fold, the 28 training donor-task control means are the inputs. Separate fits use
either stimulated-minus-control means (effect fit) or stimulated means (state
fit) as targets. The input SVD rank is four and ridge lambda is one. Thus the
core contains exactly 8 folds x 2 targets = 16 ridge fits.

CellFlow is pinned to `cellflow-tools==0.0.9`, tag commit
`03bff12e71326742ad53a0909a4a8fee36958794`, as a runtime-only dependency under
PolyForm-Noncommercial-1.0.0 for academic noncommercial execution; its source is
not vendored. Checksums for the source archive and extracted tree were recorded
before the pilot. The model uses OTFM, the executable official PBMC architecture and
optimizer, 500,000 microsteps with gradient accumulation 20, terminal checkpoint
only, and `out_of_core_dataloading=false`. Each fold-seed fit covers all four
tasks. Formal initialization seeds are
`17, 29, 43, 763929762, 85508424`, giving exactly 40 core fits. Under this fixed
version, batch ordering and training-noise RNG remain zero; these are five
initialization realizations, not independent stochastic replicates. A, B and C
conditioning scenarios are inference calls against the same checkpoint, never
refits.

## Common estimand and adapters

For observed effect vector `delta_obs` and predicted effect vector `p`, utility
is

```text
U(p, delta_obs) = -(1/5000) sum_g [ (p_g - delta_obs_g) / max(s_g, 0.1) ]^2,
delta_obs = mean(stimulated 12) - mean(Cobs),
```

where `s_g` is the training-control sample standard deviation. Larger is better
and genes have equal weight.

Every native prediction is first represented as a terminal-state mean
`yhat_m(Z)` under `Cmodel=Z`. An effect-fit ridge output is converted to a
terminal state by adding the mean of Z; state-fit ridge and CellFlow already
produce terminal states. The two frozen scoring adapters are

```text
p_m^E(Z)   = yhat_m(Z) - mean(Z)
p_m^S(Z,P) = yhat_m(Z) - mean(P)
```

and must satisfy `p_m^S = p_m^E + mean(Z) - mean(P)` numerically.

## Four primary donor-root endpoints

Stage A fixes `Cobs=A`, `Cmodel=C` and compares `P0=A` with `P1=B`. Let

```text
R_k^{ab} = U_ridge-effect-fit^a(A, P_k, C)
           - U_CellFlow^b(A, P_k, C),       a,b in {E,S}
M_k      = R_k^{ES} - 0.5 (R_k^{EE} + R_k^{SS})
H_k      = U_ridge-effect-fit^E(A, P_k, C)
           - U_ridge-state-fit^S(A, P_k, C)
Z_1d     = mean_{seed,task,permutation}(M_1 - M_0)
Z_2d     = mean_{task,permutation}(H_1 - H_0)
```

`M_k` also equals `0.5 (R_k^{ES} - R_k^{SE})`, which is an implementation
identity. E1 isolates the cross-output benchmark contrast relative to the two
same-output controls; E2 tests matched ridge effect- versus state-target fits.
E2 does not vary across model initializations.

Stage B fixes `Cobs=A`, `Cpred=B`. With CellFlow's state adapter,

```text
Q(Z)  = U_CellFlow^S(A, B, Z)
Z_3d  = mean_{seed,task,permutation}[Q(A) - Q(C)]
Z_4d  = mean_{seed,task,permutation}[Q(B) - Q(C)].
```

E3 and E4 are CellFlow utility contrasts, not ridge-minus-CellFlow rankings.
They isolate model conditioning shared with observation and prediction
references, respectively. Earlier analysis of the same resource supplied the
prespecified raw directions `(E1,E2,E3,E4)=(+,+,-,+)`; the source of these directions must be
reported.

The fixed aggregation order is: cell-to-profile means; equal-gene utility;
contrast within donor-task-seed-permutation; equal mean over six permutations;
equal mean over four tasks; equal mean over five initialization realizations for
E1, E3 and E4; retain the eight donor-root values. E2 skips the seed step.

## Working-law support and reporting

Raw root values are multiplied by their prespecified direction. The four
oriented endpoint vectors are analysed jointly using all `2^8=256` donor sign
vectors, with the same sign vector applied to all endpoints, and the frozen
recentered/restudentized max-|t| working-law procedure at familywise alpha 0.05.
The resulting intervals are sensitivity summaries only, not confidence
intervals, P values, randomization inference or donor-population inference.

An endpoint receives **primary directional support** exactly when its oriented
simultaneous lower bound is strictly greater than zero. Separately, it receives
the **uniform finite-panel robustness** label when all eight oriented donor-root
values are strictly positive. The 8/8 label is reported but is not necessary
for primary support.

E1 and E2 must both have primary support to support output-representation
attribution. E3 and E4 must both have primary support to support model-
conditioning allocation. All four must have primary support to support joint
attribution. Partial support is reported without upgrading the unsupported
mechanism. The title and abstract may name only mechanisms whose paired primary
gate passes. All four endpoints, all eight root values and every failed gate are
reported whenever the formal panel is complete.

## Execution and failure rules

One complete technical pilot uses initialization seed `20260831`, which is
disjoint from the formal seeds. Only implementation, identity, geometry,
finiteness and resource gates may be inspected; pilot predictions are discarded
and cannot change the science.

After formal execution began, failed fits were not replaced, retuned or rerun
with another seed. A failure shown to have occurred before the analysis code
started could be resubmitted once with identical inputs. Completeness required
all 40 CellFlow fits, all 16 ridge fits and finite values for every primary
endpoint; otherwise the analysis was classified as `INCONCLUSIVE_INCOMPLETE`.

The following secondary analyses did not affect the primary interpretation:
RBF ridge, unsquared energy-distance U-statistics in fold-only PCA space,
strict reversal, remaining `Cobs` factorial contrasts and Stack. CPA, scGen and
CellOT were retained as historical comparators.

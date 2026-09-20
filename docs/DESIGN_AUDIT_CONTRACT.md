# Benchmark-design audit V1 contract and evidence boundary

## Scope

V1 applies one-sided exact-sign tests with Holm adjustment to supplied
unit-level utilities under a declared benchmark design. It checks the
configuration, calculations and output files.

The configuration must specify the model order and output types, training and
evaluation separation, reference assignments and depths, overlap, aggregation,
units and independence, utility definitions, analysis timing, comparison family
and decision rule. Missing or additional fields are rejected.

`reference_declaration.mapping_scope` is fixed to
`DECLARED_BLOCK_ABSTRACTION_NO_CELL_MEMBERSHIP_VERIFICATION`. Reference checks
compare block labels, not cell membership, partial overlaps or cell counts.
Depth is recorded as a required label and unit, including when its value is
non-public or non-numeric.

## What each status means

Every valid pairwise result has exactly one `decision`:

- `DIRECTION`: all required declarations are present and their conditions are
  met, arithmetic support reaches `K_min`, and the adjusted exact-sign result
  rejects at the specified alpha;
- `NO_DIRECTION`: the same declarations and arithmetic-support conditions are
  met, but the adjusted result does not reject;
- `WITHHELD`: a required declaration is missing or a condition is not met, with
  the cause in `withheld_reason`.

V1 emits `UNIT_INDEPENDENCE_NOT_ESTABLISHED`,
`TRAINING_EVALUATION_SEPARATION_NOT_ESTABLISHED` and
`ARITHMETIC_SUPPORT_BELOW_KMIN` as withheld reasons. An optimizer-based audit
could use `OPTIMIZATION_CERTIFICATION_UNCERTAIN`, but this deterministic exact
evaluator performs no optimization search and never emits that reason.

A missing required declaration is an invalid configuration, and malformed or
incomplete utility support is an input-contract error. Those cases return a
nonzero CLI status and create no evaluation bundle. An explicit
`NOT_ESTABLISHED` independence status is valid: the arithmetic is retained for
inspection, but `reject_at_alpha` is `NA` and all decisions are `WITHHELD`.

`analysis_timing` is checked for internal consistency but does not affect the
decision. `DIRECTION` reports the exact-sign/Holm result conditional on the
supplied declarations; preregistration, timing of choices and external protocol
status require evidence outside this check.

## Machine checks versus assertions

| Item | Machine-checked fact | Remaining assertion or limitation |
|---|---|---|
| Configuration | Required fields, allowed values, unique ordered identities and byte-identical snapshots | The implementation and terminal-state strings correspond to the code or fitted object that produced utilities |
| Output semantics | Each configuration declares `DIRECT_EFFECT` or `ABSOLUTE_STATE` | The emitted model object and reference conversion actually have that meaning |
| Reference construction | All three role-to-block labels exist; declared overlap equals label-derived overlap; depth labels and aggregation rule are present | Cell membership, numeric depth, block disjointness in source data and upstream aggregation execution |
| Training/evaluation separation | Status, basis and authority are present and internally valid | Whether the declaration matches actual data reuse; the schema requires `machine_verified=false` |
| Unit geometry | Metadata hierarchy, object uniqueness, within-unit consistency, multiplicity and candidate label geometry | Candidate components and absence of recorded links do not prove independence |
| Independence | Status, basis and authority are present; evaluation consumes the bound unit-audit record | Scientific independence; the schema requires `machine_verified=false` |
| Utility support | Source hash when pinned, column meanings and constants, finite values, exact audited-unit agreement and complete unit × method support | Upstream cells, scores, predictions, membership, checkpoints and utility construction |
| Timing | Required timing label and simple internal consistency of result-access fields | Whether the timing declaration matches the analysis history or an external protocol |
| Family and rule | Complete ordered pairs, orientation, literal-zero ties, exact fair-binomial tails, Holm scope and `K_min` | Whether the declared family was scientifically complete or prospectively fixed |
| Output integrity | Closed file tree, SHA-256 bindings, canonical receipts and separate semantic recomputation | Authenticity and accuracy of the supplied declarations |

The output includes the pseudonymized unit metadata, unit-audit receipt and
manifest, allowing it to detect configuration changes between stages.

## Public empirical example

`analysis/design_audit/public_empirical_example/` reads the separately distributed
192-row `ROOT_01`–`ROOT_08` utility table. It checks the table hash, declared
column constants, all eight roots, the 168-row ordered comparison family and
output files. The example starts from supplied utilities; real donor/batch
labels and upstream cell/model records are not included.

The paper's separate public replay has 84 unordered contrasts and uses
max-|T|, Bonferroni-t and marginal-t/Holm sensitivities. V1's generic audit
uses 168 ordered one-sided exact-sign/Holm directions in three 56-direction
families. The empirical example therefore audits supplied root-level
utilities; it does not reproduce the primary decision rule.

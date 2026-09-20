# Public utility-level example

This example passes the separately distributed
`GSE162632_ROOT_UTILITIES_PUBLIC_V1.tsv` table through the generic
benchmark-design audit. The only unit identifiers are local
`ROOT_01`–`ROOT_08`. The metadata and reference-block labels are pseudonymized.

The audit starts from 192 pseudonymized root utilities. It checks table schemas
and agreement between configuration, output and unit declarations. The cells,
memberships, predictions and fitted states used to produce these utilities are
not distributed with this example, so those upstream steps are not checked.

The generic CLI and the primary public replay use different
decision rules:

- this generic audit plans 168 ordered directions (three allocations × 56)
  and applies one-sided exact-sign tests with Holm adjustment separately
  within each 56-direction allocation family;
- `scripts/build_gse162632_public_evidence_v1.py --check` reproduces the
  paper's 84 unordered within-scheme contrasts and its max-|T|,
  Bonferroni-t and marginal-t/Holm sensitivities.

The generic family requires at least `K_min=11` non-tied roots. An eight-root
input therefore yields 168 `WITHHELD` decisions with reason
`ARITHMETIC_SUPPORT_BELOW_KMIN`, even though the supplied declaration states
that the roots are independent. The separate
84-contrast replay uses a different decision rule.

From the repository root, write and verify all three generic bundles in a new
directory:

```bash
python analysis/design_audit/public_empirical_example/run_public_empirical_example.py \
  --source-data-root /path/to/05_SOURCE_DATA \
  --output-root /path/to/new/audit-run
```

Each bundle contains its configuration, run receipt and SHA-256 manifest.
The evaluation bundle also includes the pseudonymized unit metadata and unit
audit used to decide whether directional testing is permitted.

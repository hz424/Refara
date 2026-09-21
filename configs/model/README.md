# Model configuration scope

To recalculate the historical 84-contrast family and the separate post hoc
56 paired construction changes, obtain the matching Source Data package and
run these commands from the repository root. They use its 192-row root-utility
table and the implementation in
`src/perturb_nuisance_contracts/root_family_inference.py`:

```bash
python scripts/build_gse162632_public_evidence_v1.py \
  --check --source-data-root /path/to/05_SOURCE_DATA
python scripts/build_gse162632_paired_construction_changes_public_v1.py \
  --check --source-data-root /path/to/05_SOURCE_DATA
```

The table's `public_root_code` column uses pseudonyms `ROOT_01` through
`ROOT_08`; the mapping to study identifiers is not included.

This directory contains generic configuration material. Rebuilding the full
study prediction panel requires the original production configuration, data
locators, membership records, selection salts, score inputs, provider data,
fitted models and runtime, which are outside this release.

For a replay using bundled arrays, start with the
[focal scGen capsule](../../capsules/gse162632_scgen/README.md). Its CPU path
recalculates scores; optional retraining requires the separate prepared-data
archive.

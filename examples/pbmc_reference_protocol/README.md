# PBMC predictions with two reference blocks

This example scores bundled GSE162632 predictions from PCA-64 ridge and scGen
under a fresh-reference effect target. It uses the focal capsule's q1 realization
and disjoint depth-eight reference construction: 40 tasks from eight donors on
a fixed axis of 2,000 pseudonymized features.

After [installing the repository](../../README.md#install), run from its root:

```bash
.venv/bin/python examples/pbmc_reference_protocol/run.py --output outputs/pbmc_protocol
```

Choose a new output directory. The command runs on CPU using bundled arrays.

Open `outputs/pbmc_protocol/results/report.md` for the comparison. The output also
contains `plan.json`, exported TSV inputs under `inputs/`, the score tables under
`results/`, and `verification.json`. The script checks the source files against
the capsule manifest, then compares 80 task scores, 16 donor scores and the overall
pairwise margin with the existing focal results. `verification.json` records the
maximum absolute error and the comparison tolerance.

B1 supplies prediction centring and B2 defines the observed effect. The primary
comparison therefore uses two blocks. The plan requests the optional balanced
five-pattern diagnostic; the report records that three blocks are needed for that
calculation, while retaining the primary scores.

PCA-64 ridge supplies a native effect. scGen remains a state model in the plan,
even though its cached output is exported as an effect using the capsule's original
float32 subtraction. The original baseline accompanies this export and restores
the state before scoring. This preserves the arithmetic of the focal replay.

The saved scGen predictions retain their original model inputs; B1 supplies
only the scoring baseline in this example. The mean arrays and their hashes
identify the supplied values, but checking which cells the model consumed
requires the original provenance and membership records.

Tasks are averaged equally within each donor and donors equally overall. The
result is a descriptive replay of the supplied predictions. See
[choosing references](../../docs/choosing-references.md) for selecting a target
and [the reference guide](../../docs/REFERENCE_DESIGN.md) for using your own files.

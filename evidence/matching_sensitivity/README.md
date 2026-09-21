# Matching sensitivity: numerical replay

Recalculate directional tests and leading models for alternative maximum
donor–batch matchings using the supplied matching summaries.

From the repository root, using Python 3.10 or newer (standard library only):

```sh
python evidence/matching_sensitivity/replay.py --output outputs/matching_sensitivity
```

The output directory must be new. The script writes four TSV tables and
`SUMMARY.json`:

- `directional_decisions.tsv`: recomputed exact one-sided sign-test tails and
  Holm-adjusted values for all 56 ordered model comparisons in each of ten
  recorded maximum-matching witnesses. Tied observations are excluded from
  the sign-test denominator; the full 56-member family is retained.
- `directional_sensitivity.tsv`: comparisons with both rejecting and
  nonrejecting witnesses at 0.05. The supplied data demonstrate RBF versus TW
  and RBF versus PCA for G39, and RBF versus CM for G12.
- `matching_leaders.tsv`: exact mean utility and rank for each of eight models
  in each of 144 G12 maximum matchings. All exact ties are retained as leaders.
- `leader_counts.tsv`: counts recomputed from those ranks. NC leads in 67
  matchings and RBF in 77; the remaining models lead in none.

`source/witness_counts.tsv` contains only the sign counts used by the tests.
`source/g12_matching_utility_sums.tsv` contains eight utility sums for each
G12 matching, encoded as exact numerator–denominator pairs derived from the
stored binary64 utilities. The replayer divides these sums by the matching
cardinality of 12 and calculates ranks and leader counts.
`BINDINGS.json` authenticates both inputs. `PROVENANCE.json` traces their sources.

The replay starts from the ten verified witness summaries and 144 enumerated
maximum matchings. Reconstructing the donor–batch graph, enumerating matchings,
generating wave utilities and checking all-graph directional certificates
require the upstream inputs.
G39 denotes the recorded graph with maximum matching size 39; G12 denotes the
training-batch-excluded graph with maximum size 12. Donor and batch identifiers,
graph edges, memberships and per-edge utilities are excluded.

The sign-test interpretation remains conditional on its declared sampling
model; donor–batch nonreuse alone does not establish residual independence.
Holm correction applies within each recorded matching's 56-direction family.
The matching counts describe a finite enumeration, not probabilities of
selecting a model or population-level superiority.

These summaries derive from the Allen Institute Sound Life resource and retain
its provider terms, including the research/noncommercial scope. See
[licence and attribution](LICENSE_SCOPE.md).

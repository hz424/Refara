# Sound Life robust maximum-matching analysis V2

## Analysis timing

This specification was written after the V7 results and the V1 salted matching
ensemble were available. The analysis is therefore post hoc. The specification
was frozen before production V2 artifacts entered the manuscript package, but
it is not a preregistration and the results remain a sensitivity analysis.

The donor–batch graphs, candidate edges and maximum cardinalities are
reconstructed and hashed before the utility table is examined. Utilities are
then used only to identify extremal matchings within the fixed graphs.

## Fixed graph family

- SAME_BATCH_SAME_POOL: 93 candidate donor-wave edges, 58 donors, 40 batches,
  maximum cardinality 39.
- SAME_BATCH_SAME_POOL_TRAIN_BATCH_EXCLUDED: 23 candidate donor-wave edges,
  22 donors, 12 batches, maximum cardinality 12.

The first graph is handled by exact additive optimization over all
maximum-cardinality matchings. The second graph is exhaustively enumerated;
the expected count is 144.

## Fixed outputs

For each graph:

- edge inclusion minimum and maximum over all maximum matchings;
- minimum and maximum equal-wave mean utility for every configuration;
- minimum and maximum mean contrast for all 56 ordered pairs;
- minimum and maximum win and literal-tie counts for all ordered pairs;
- whether each configuration can be a descriptive mean leader;
- whether each configuration must be a descriptive mean leader;
- whether each working Holm-threshold conclusion is invariant, variable,
  impossible, or unresolved.

For the 12-wave graph, leader and Holm classifications are exhaustive.

For the 39-wave graph, additive bounds use an exact rational min-cost
maximum-flow implementation. Possible-leader feasibility uses a fixed
mixed-integer maximin problem and is independently checked against the exact
binary-rational utility values of the returned membership. Necessary-leader
status uses exact lower bounds for all seven pairwise mean margins.

Holm robustness for the 39-wave graph uses only logically sufficient
certificates:

- invariant rejected when the worst-case raw p-value passes the complete
  56-test Bonferroni bound;
- impossible rejection when the best-case raw p-value exceeds alpha;
- possible but not necessary when explicit rejecting and non-rejecting
  maximum-matching witnesses both exist;
- unresolved when these certificates do not decide possibility.

The 256 SHA-256 salted matchings are retained only as membership witnesses.
Their frequencies are not probabilities.

## Numerical definitions

- Frozen utilities are parsed from their hexadecimal binary64 serialization.
- Additive utility objectives are represented as exact Python Fraction values.
- A descriptive leader maximizes the exact rational equal-wave utility sum.
- Literal ties are exact equality of the frozen binary64 values.
- Raw tests are exact one-sided fair-binomial upper tails conditional on the
  declared sign model.
- Holm correction is applied within all 56 ordered comparisons at alpha=0.05.

## Interpretation

The outputs characterize sensitivity over declared maximum matchings of
recorded donor–batch graphs. They do not prove residual independence, define
a biological sampling distribution, establish a population-best
configuration, or validate the original analysis protocol retroactively.

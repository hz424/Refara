# CPU focal replay

This bundle recomputes the focal PCA-64 ridge minus scGen contrast across five training realizations. The `q1` arrays correspond to the realization used in the primary analysis; `q2` through `q5` contain four additional realizations.

The replay forms observed effects as treated mean minus observed-reference mean and scGen effects as native prediction minus prediction-reference mean. It reports standardized mean squared error and its negative utility. Tasks are averaged within each root, and the eight root values are then averaged equally. For the rotation construction, each rotation is scored before the three utilities are averaged.

`replay_cpu.py` requires only Python and NumPy. It verifies every payload hash in `manifest.json`, recalculates the focal contrasts and checks them against the reported values in `focal_contrasts.tsv`.

Any redistribution of the underlying third-party data remains subject to its source terms.

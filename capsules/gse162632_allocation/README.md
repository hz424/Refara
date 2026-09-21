# GSE162632 reference-allocation replay

Recompute the GSE162632 matched-depth reference-allocation analysis from
1,000 allocation labels, three reference depths, 27 role assignments, eight
public root indices and eight configurations. The replay calculates the five
role-overlap pattern means and all 28 pairwise allocation interactions, then
compares them byte-for-byte with the saved arrays. Supplying Source Data also
checks the GSE162632 rows used in the original Figure 2b,c.

Run from the repository root:

```bash
python capsules/gse162632_allocation/replay.py

python capsules/gse162632_allocation/replay.py \
  --source-data-root /path/to/05_SOURCE_DATA \
  --receipt /path/to/GSE162632_ALLOCATION_REPLAY_RECEIPT.json
```

The 1,000 labels specify deterministic allocations of the same cells; they are
not biological replicates or draws from a natural probability distribution.
The replay starts from scored utilities. Raw expression, cell identifiers,
membership lists, predictions, checkpoints, selection salt and private donor
mappings are excluded. For Parse, use the label-averaged Source Data tables;
provider terms limit distribution of its label-level arrays.

The Python replay code is BSD-3-Clause. The bundled pseudonymized numerical
derivatives and this documentation are released under CC BY 4.0; this licence
does not extend to the underlying GSE162632 data.

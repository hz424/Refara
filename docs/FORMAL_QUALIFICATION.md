# Qualification of the generic inference procedure

The generic reference-aware inference procedure failed its prespecified
simultaneous-coverage requirement in two of 198 scenarios, each tested with
10,000 repetitions. The recorded status is
`NO_GO_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY`. Its bounds therefore remain
diagnostics rather than calibrated confidence intervals. The paper's empirical
rank reversals are calculated directly from the evaluated scores; this
qualification concerns the additional inference procedure.

After installing `requirements/locked.txt`, run the full calculation with an
explicit worker count and a new output directory:

```bash
python scripts/replay_formal_qualification.py \
  --workers 52 \
  --output-root /path/to/new/qualification-replay
```

The original run used 52 workers; any positive worker count is accepted.
The replay comparison ignores only that field and requires every scientific
result to match. This resource-intensive calculation is excluded from the quick
demo and continuous integration.

A successful replay means the failed qualification was reproduced. The replay
command returns 0 after matching the recorded report; the underlying scientific
calculation retains its failure status and scientific exit code 2 in the receipt.

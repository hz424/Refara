# Qualification of the generic inference procedure

The generic reference-aware inference procedure failed its simultaneous-coverage
requirement in two of 198 scenarios, each tested with 10,000 repetitions
(`NO_GO_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY`). The replay reproduces this
failed qualification. This procedure is separate from the paper's empirical
rank-reversal analysis.

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

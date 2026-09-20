# Reproduce the numerical results

Start with the [current PBMC replay](current_submission/README.md) for Figures
2–3 and Extended Data Figure 9, including the completed 2,560-epoch CPA fits.

The [organizer workflow](../docs/organizer.md) reproduces the Norman reporting
rules and compares them with a development-calibrated margin threshold.

The six-component suite below covers the other study analyses and preserves
the earlier 1,280-epoch CPA model-comparison and Figure 3 results. From the
repository root, run:

```bash
.venv/bin/python evidence/reproduce.py --output outputs/conclusions
```

The command writes a `RESULTS.json` summary and output tables for each component.
The [reproduction guide](../docs/REPRODUCING.md) lists the inputs and calculations.

| Component | Main calculation |
|---|---|
| `reference_roles/` | Direct residuals, the scoped sharing identity and invariance controls |
| [model_comparisons/](model_comparisons/README.md) | Seed/unit aggregation, pair margins, ranks and reversals |
| [figure3/](figure3/README.md) | Input-induced displacement and score comparisons |
| [program_responses/](program_responses/README.md) | Program disagreements and observation-specific prediction errors |
| [matching_sensitivity/](matching_sensitivity/README.md) | Directional decisions and leaders across supplied matching summaries |
| `replay_table1.py` | Single-role score sensitivity and cached-array operations |

## Role sensitivity

```bash
.venv/bin/python evidence/replay_table1.py --output outputs/role_sensitivity
```

This command recalculates summaries from the Norman task-score cube and runs
95 operations on saved prediction arrays. It includes all six depths and four
metrics, with checks for six input-conditioned CPA role/depth values and ten
fixed-output input zeros. `TABLE1_CURRENT_SELECTION.json` identifies the
manuscript values.

The statistic averages signed changes across 55 tasks before taking absolute
values, then averages reassignments and allocations. Scoring-to-input ratios
are ratios of these summaries. The original CPA configuration retains its saved
outputs; input-conditioned CPA selects predictions generated from the new input
block. They use different fits, with parameters fixed during each audit.

The archived `table1/` companion is verified by file hashes. Write generated
outputs outside `evidence/` to keep its inputs intact.

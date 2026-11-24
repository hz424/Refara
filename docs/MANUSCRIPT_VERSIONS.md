# Manuscript versions and display-item numbers

The guides in this repository use the figure and table numbers of the current
manuscript, *Control-cell references can decide whether deep models outperform
baselines in single-cell perturbation benchmarks*.

Release 1.10.0 accompanied an earlier version of the same study, titled
*Reference-cell design shapes model evaluation in single-cell perturbation
prediction* (September 2026). The numerical evidence it reproduced is unchanged.
Its display items map to the current manuscript as follows.

| September 2026 version | Current manuscript | Content |
|---|---|---|
| Figure 1 | Fig. 1, panels a and b | Control roles and effect scoring (the homepage overview) |
| Figure 2 | Fig. 2 | Reassigning control cells reverses model rankings |
| Figure 3 | Fig. 5 | Changing model inputs |
| Figure 4 | Fig. 6 | Program predictions after reference-dependent selection |
| Figure 5 | Supplementary Fig. 14 | Alternative donor–batch matchings |
| Extended Data Figure 1 | Supplementary Fig. 1 | Rankings under standardized MSE and delta-RMSE |
| Extended Data Figure 2 | Supplementary Fig. 4 | Program-interpretation sensitivities |
| Extended Data Figure 3 | Supplementary Fig. 5 | Genetic-perturbation prediction (Norman) |
| Extended Data Figure 4 | Supplementary Fig. 6 | Simulations of replication and test choice |
| Extended Data Figure 5 | Supplementary Fig. 7 | Acquisition geometry |
| Table 1 | Table 1 | Score sensitivity to reference reassignment |
| Supplementary Tables 1–9 | Supplementary Tables 1–9 | Unchanged numbers |

New in the current manuscript: Fig. 1 panel c, Fig. 3, Fig. 4 and Supplementary
Tables 10 onward. The code for Fig. 3, Fig. 4 (panels a and b) and Supplementary
Tables 11–16 is in
[`analysis/control_reference_noise`](../analysis/control_reference_noise/README.md).

## Labels that were not renamed

Directory names, file names, output keys and comments in code are stable
identifiers of the numerical evidence and keep the labels they were created
with. Examples are `evidence/figure3/` (Fig. 5 of the current manuscript),
`ed9/` (Supplementary Fig. 1) and file names containing `EXTENDED_DATA_FIGURE`
or `Figure_4`. A number inside a path or an output key is therefore not a
current display number. Where such a label appears, the guide beside it states
which current item it belongs to; documents that describe an "original",
"earlier" or "historical" figure refer to the numbering in use when that
analysis was first archived.

The guide inside [`evidence/current_submission/`](../evidence/current_submission/README.md)
is one of the files that replay verifies by checksum before it runs
(`evidence/current_submission/MANIFEST.json`), so it is unchanged and uses the
September 2026 numbers: its Figure 3, Figure 4 and Extended Data Figure 1 are
Fig. 5, Fig. 6 and Supplementary Fig. 1 of the current manuscript. Its Figure 2
is Fig. 2.

The supplementary figures of the September 2026 version are not cited by number
in this repository.

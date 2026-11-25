# Manuscript versions and display-item numbers

The guides in this repository use the figure and table numbers of the current
manuscript, *Reference-cell design can reverse model rankings in single-cell
perturbation prediction* (November 2026 version).

Release 1.10.0 accompanied the September 2026 version of the same study, titled
*Reference-cell design shapes model evaluation in single-cell perturbation
prediction*. The numerical evidence it reproduced is unchanged, with Influenza
PBMC CPA unified to the 2,560-epoch training-selected refits across all analyses.
Its display items map to the current manuscript as follows.

| September 2026 version | Current manuscript | Content |
|---|---|---|
| Figure 1 | Fig. 1, panels a and b | Control roles and effect scoring (the homepage overview) |
| Figure 2 | Fig. 5, panel a | Reassigning finite control blocks reverses model rankings |
| Figure 3 | Supplementary Fig. 15 | Changing model inputs at fixed scoring references |
| Figure 4 | Supplementary Fig. 16 | Program predictions after reference-dependent selection |
| Figure 5 | Supplementary Fig. 14 | Alternative donor–batch matchings |
| Extended Data Figure 1 | Supplementary Fig. 1 | Rankings under standardized MSE and delta-RMSE |
| Extended Data Figure 2 | Supplementary Fig. 4 | Program-interpretation sensitivities |
| Extended Data Figure 3 | Supplementary Fig. 5 | Genetic-perturbation prediction (Norman) |
| Extended Data Figure 4 | Supplementary Fig. 6 | Simulations of replication and test choice |
| Extended Data Figure 5 | Supplementary Fig. 7 | Acquisition geometry |
| Table 1 | Supplementary Table 23 | Score sensitivity to reference reassignment |
| Supplementary Tables 1–9 | Supplementary Tables 1–9 | Unchanged numbers |

New in the current manuscript: Fig. 1 panel c, Fig. 2 (released STATE checkpoints under squared error and `cell-eval2` v0.22.0, including the six-metric challenge composite score and five-allocation stability; Supplementary Tables 18, 22 and 24–26), Fig. 3 (reanalysis of deposited predictions across four CRISPR perturbation screens; Supplementary Tables 21 and 27), Fig. 4 (own-control centring, uncapped full-pool PBMC fits up to 3,270,423 training cells, and evaluation across all 375,241 Parse control cells; Supplementary Tables 15–17, 19–20 and 28), and Fig. 5 panels b and c (reference-noise correction and cross-individual reference borrowing across capped and uncapped fits up to all 375,241 Parse control cells; Supplementary Tables 10–14 and 28).

The scripts for reference-noise, disjoint-cell, cross-individual and own-control
analyses are in [`analysis/control_reference_noise`](../analysis/control_reference_noise/README.md),
and the paired-shift and ranking-reversal script for the four CRISPR benchmarks is in
[`analysis/published_crispr_benchmarks`](../analysis/published_crispr_benchmarks/README.md);
the remaining benchmark and figure scripts accompany the manuscript submission.

## Labels that were not renamed

Directory names, file names, output keys and comments in code are stable
identifiers of the numerical evidence and keep the labels they were created
with. Examples are `evidence/figure3/` (Supplementary Fig. 15 of the current
manuscript), `ed9/` (Supplementary Fig. 1) and file names containing
`EXTENDED_DATA_FIGURE` or `Figure_4`. A number inside a path or an output key
is therefore not a current display number. Where such a label appears, the
guide beside it states which current item it belongs to; documents that
describe an "original", "earlier" or "historical" figure refer to the numbering
in use when that analysis was first archived.

The guide inside [`evidence/current_submission/`](../evidence/current_submission/README.md)
is one of the files that replay verifies by checksum before it runs
(`evidence/current_submission/MANIFEST.json`), so it is unchanged and uses the
September 2026 numbers: its Figure 2, Figure 3, Figure 4 and Extended Data
Figure 1 are Fig. 5a, Supplementary Fig. 15, Supplementary Fig. 16 and
Supplementary Fig. 1 of the current manuscript.

The supplementary figures of the September 2026 version are not cited by number
in this repository.

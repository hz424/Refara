# Refara

Refara audits how control-cell reference design affects model comparisons in
single-cell perturbation prediction. It separates changes to model inputs from
changes to scoring references, compares designs at a fixed control-cell budget,
and identifies comparison directions that persist across specified allocations.

This repository accompanies *Reference-cell design shapes model evaluation in
single-cell perturbation prediction*.

![Control-cell roles in model prediction and evaluation.](docs/assets/figure-1.png)

[How reference roles affect model comparisons](docs/overview.md)

<a id="install"></a>

## Quick start

Use **Python 3.10.x**. The full environment was tested on Linux x86-64 with
Python 3.10.21. From the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --no-deps -r requirements/vcc-build-py310.txt
.venv/bin/python -m pip install --no-deps -r requirements/vcc-py310-linux.txt -r requirements/unified-prepare-py310.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/reference-design audit demo --output outputs/reference_audit
```

The CPU demo uses bundled synthetic data and writes `AUDIT.json`. A successful
run has status `PASS_EXECUTED_SYNTHETIC_OPERATION_AUDIT`. Use a new output
directory for each run. A cached Linux installation took about two minutes;
the demo took less than one second on an Intel Xeon Gold 6230R with two threads.

## Use your own predictions

Start by [choosing references](docs/choosing-references.md), then
[prepare AnnData or TSV inputs](examples/anndata_prepare/README.md).

| Task | Guide or example |
|---|---|
| Score predictions under a declared reference design | [Scoring guide](docs/REFERENCE_DESIGN.md) |
| Change model inputs or rescore saved predictions | [Reference operations](docs/REFERENCE_AUDIT.md) |
| Run a trained model on new control allocations | [Control-dependent scGen](examples/conditioned_model_reference/USAGE.txt) · [Direct-effect PCA](examples/real_model_reference/USAGE.txt) |
| Compare control budgets and reporting rules | [Benchmark organizer](docs/organizer.md) |
| Compare five reporting rules at matched coverage | [Reporting comparison](evidence/reporting_reliability/USAGE.txt) |
| Audit official cell-eval2 metrics | [VCC adapter](docs/VCC_AUDIT.md) · [Python 3.12 scorer setup](docs/VCC_INSTALLATION.md) |

The model examples include normalization, inference and S/O scoring from a small
count matrix. Both check that equivalent effect and state representations receive
the same O score. The scGen example also separates input changes from rescoring.

## Reproduce the paper's results

Run the current PBMC numerical replay after installation:

```bash
.venv/bin/python evidence/current_submission/replay.py --output outputs/current_submission
```

It reconstructs the numerical summaries for Figures 2 and 3 and Extended Data
Figure 1 from saved scores and summaries for each fitted seed and evaluation
donor, including the 2,560-epoch Influenza CPA fits. It writes the results and
checks to `RESULTS.json`.

The [reproduction guide](docs/REPRODUCING.md) lists the remaining analyses,
commands and required inputs. [Data access](docs/DATA_ACCESS.md) covers public
sources and additional reconstruction files.

## Interpret the audit

Allocation stability describes the evaluated data and control assignments.
Inference across experiments depends on the experimental units and statistical
procedure, as explained in the [statistical guide](docs/STATISTICAL_INTERPRETATION.md).

The [retrospective reporting comparison](docs/REPRODUCING.md#five-reporting-rules-on-parsegse181897)
found no advantage for Refara at matched coverage. In the
[external GSE181897 analysis](docs/REPRODUCING.md#external-fixed-budget-comparison),
shared and observation-separated selection both chose PCA and produced identical
predictions for 46 assessment donors. These results distinguish sensitivity to
reference design from improved predictive performance.

## Documentation and development

[All guides](docs/README.md) · [Contributing](CONTRIBUTING.md) ·
[Citation](CITATION.cff)

Run `.venv/bin/python -m pytest -q` for the test suite. Tests requiring external
Source Data are skipped when those files are absent.

The [v1.10.0 release](https://github.com/hz424/Refara/releases/tag/v1.10.0)
contains a wheel, the complete source ZIP and checksums. Examples and numerical
replays are in the source ZIP. Author-written code is BSD-3-Clause; documentation
and the overview figure are CC BY 4.0. Data retain their
[source-specific terms](LICENSE_SCOPE.md).

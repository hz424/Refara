# Refara documentation

Start with the [demo](../README.md#quick-start), then
[choose references](choosing-references.md) for your prediction task.

## Use Refara

| Guide | What you can do |
|---|---|
| [Choose references](choosing-references.md) | Decide which controls to use for prediction and scoring |
| [Prepare AnnData or TSV inputs](../examples/anndata_prepare/README.md) | Prepare your data and predictions |
| [Scoring](REFERENCE_DESIGN.md) | Write a scoring plan and read the results |
| [Change references](REFERENCE_AUDIT.md) | Work out whether to generate new predictions or rescore existing ones |
| [Control-dependent scGen example](../examples/conditioned_model_reference/USAGE.txt) | Run a trained encoder and decoder to separate input changes from rescoring |
| [Direct-effect PCA example](../examples/real_model_reference/USAGE.txt) | Normalize counts, predict effects and check equivalent state scores |
| [Eight-donor PBMC example](../examples/pbmc_reference_protocol/README.md) | Run a complete example |
| [Statistical interpretation](STATISTICAL_INTERPRETATION.md) | Distinguish control-sampling variation from biological replication |
| [Control cells and model comparisons](overview.md) | Understand the homepage figure |

## Compare models in a benchmark

| Guide | What you can do |
|---|---|
| [VCC adapter](VCC_AUDIT.md) | Examine reference use with the official scorer |
| [VCC installation](VCC_INSTALLATION.md) | Set up the scoring environment |
| [Organizer workflow](organizer.md) | Compare control budgets and reporting choices |
| [Direction reporting](organizer.md#summarize-independently-assessed-directions) | Summarize supported, reversed, unresolved and held comparisons from supplied intervals |
| [Fair comparisons](FAIR_COMPARISON_GUIDE.md) | Match predictions, inputs and evaluation targets |

## Reproduce the paper and contribute

| Guide | What you can do |
|---|---|
| [Reproduce the results](REPRODUCING.md) | Find commands, inputs and expected results |
| [Data access](DATA_ACCESS.md) | Locate additional data and their terms of use |
| [Upstream reconstruction](../reconstruction/current_upstream/README.md) | Check required files and run native CPA continuation or prediction |
| [Additional analyses](ADDITIONAL_ANALYSES.md) | Run simulations and training reviews with their required inputs |
| [Contributing](../CONTRIBUTING.md) | Report a bug or test a change |

## Repository layout

| Directory | Contents |
|---|---|
| `src/`, `specs/`, `configs/` | Toolkit, input schemas and configurations |
| `examples/`, `tests/`, `tests_official/` | Examples and tests |
| `evidence/` | Inputs and scripts for reproducing the main results |
| `reconstruction/` | Upstream input manifests and native model commands |
| `analysis/`, `capsules/`, `simulations/` | Individual analyses, replay datasets and simulations |
| `requirements/`, `scripts/`, `external/` | Software dependencies, command helpers and external resources |
| `data/`, `provenance/` | Data access, attribution and source records |

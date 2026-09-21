# Program-summary replay

This entry reproduces the GSE162632 program summaries, reference sensitivities, gene-overlap tables and response-magnitude tables from the supplied cached inputs. It reuses the eight fitted configurations and makes no model calls.

The replay has two analysis stages. First, `biological_consequence.py` regenerates the frozen 40-task results. Second, `summarize_program_interpretation.py` writes `results/interpretation/`, a post hoc descriptive decomposition of those unchanged program scores. The decomposition reports direction changes across four score thresholds, changes in top-*k* program membership, alignment with measured directions and sensitivities across 30 prediction-reference settings. These summaries do not treat the five cell-type tasks within a donor or overlapping Hallmark programs as independent replicates.

Use Python 3.10.17 with the exact dependencies in `requirements.txt`, including NumPy 1.26.4. From this directory:

```bash
python replay.py --source-data-root /path/to/Source_Data --output-dir /path/to/new-output
```

The command validates the source component, regenerates both analysis stages, compares their files with the recorded outputs and runs the independent arithmetic audit. Output directories must be new. The representative-case files remain as unplotted provenance.

`Source_Data` may instead be named `05_SOURCE_DATA`. It must contain the complete `Biological_consequences_v189` component and the unchanged Kang source files. `BINDINGS.json` records the exact code, input and numerical-result digests.

The analysis specification and first-run provenance remain with Source Data. Each replay writes a new run receipt while requiring exact scientific and descriptive output files. The standalone Source Data validator can also be run without this code directory.

Run the synthetic implementation checks with:

```bash
python -m unittest -v test_biological_consequence.py
```

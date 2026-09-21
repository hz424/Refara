# Data access and redistribution

Numerical replays use bundled synthetic examples, pseudonymized arrays and score
tables. Reconstructing earlier stages requires the matching Source Data package
or original study inputs. [Licence scope](../LICENSE_SCOPE.md) identifies the
applicable source terms. Source Data directory names retain historical figure
numbers; use the [reproduction guide](REPRODUCING.md) for the current figure mapping.

| Analysis | Available inputs and access |
|---|---|
| Norman genetic-combination prediction | Full reconstruction uses the original GSE133344 expression matrix. Bundled inputs include scored utilities, selection summaries and pair tables; the selected-CPA component also includes its checkpoint and fixed-panel arrays. Reviewer Source Data supplies normalized inputs, fitted models and gene-level predictions. Original dataset terms apply. |
| GSE162632 held-out analysis | [GEO GSE162632](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162632) supplies the study data. Source Data includes pseudonymized utilities, cached predictions, control and treated means, gene identifiers and deterministic allocation indices. |
| GSE162632 39-donor follow-up | `Biological_validation_v1810/` in Source Data contains deidentified donor, batch and capture-orientation summaries from the earlier measured-response test. `evidence/program_responses/` contains saved donor errors for the later selected-model comparison; `evidence/organizer/transfer/` contains reporting scores and aggregate membership checks. These donors were excluded from fitting and candidate screening but share the study and experimental batches with the discovery data. |
| GSE181897 external model-selection analysis | [GEO GSE181897](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181897) provides the original study data. `evidence/gse181897_reference_design/` bundles the protocol, model contract and saved task losses for 16 selection donors and 46 assessment donors. The replay starts from these losses; expression preprocessing and model inference require the original inputs. |
| Kang conditioning analysis | Four locally authored files in `Extended_Data_Figure_2_attribution/` provide summaries for eight evaluation donors: shifts, five-endpoint diagnostic bounds, utility responses and standardized displacement. These files exclude donor correspondence, expression matrices and study-wide predictions. |
| Parse PBMC analyses | Obtain expression data from the [Parse dataset page](https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/). Parse-derived and mixed-resource tables listed in [data/derived/LICENSE_SCOPE.md](../data/derived/LICENSE_SCOPE.md) retain CC BY-NC 4.0 noncommercial-use terms. |
| Jerber line–pool audit | The [study](https://doi.org/10.1038/s41588-021-00801-6) and [processed-data record](https://zenodo.org/records/4651413) provide the iPSC data and line–pool metadata. Separate Source Data contains an aggregate figure input. |
| GSE306429 metadata audit | Study data and acquisition metadata are available from [GEO GSE306429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306429). Aggregate audit and figure tables are distributed separately. |
| Sound Life matching | Obtain authorized individual-level inputs from the [Dynamics of Human Immune Health and Age download page](https://apps.allenimmunology.org/aifi/insights/dynamics-imm-health-age/downloads/scrna/) under the [Allen Institute Terms of Use](https://alleninstitute.org/legal/terms-of-use). Source Data contains aggregate summaries; graph-level inputs require the separate review attachment below. |

<a id="reproduction-boundary"></a>

## What you can reproduce

Saved scores support reaggregation; prediction and reference arrays support
recalculation of residuals. The biological-consequence replay uses saved
predictions, reference means and fixed Hallmark definitions from Source Data.
The focal capsule bundles prediction and reference arrays; optional scGen
retraining requires its separate prepared-data archive and a GPU.

The [reproduction guide](REPRODUCING.md) gives commands and starting inputs.
For cell selection, preprocessing and other model fitting, the
[upstream guide](../reconstruction/current_upstream/README.md) lists required
assets, hashes, local path configuration and native CPA continuation/prediction
commands. Source access conditions still apply. Manuscript artwork and drawing
scripts are supplied separately, except for the homepage Figure 1 overview.

## Additional reconstruction files

The following archives are distributed separately from this repository:

- `SOURCE_MEMBERSHIP_REVIEW_V189.zip`: links the labels used in the eight-donor
  GSE162632 summaries (`ROOT_01`–`ROOT_08`) to the original donor and cell records
  and supplies metadata for reconstruction.
- `Sound_Life_Graph_Utilities_for_Review.zip`: graph, weights and utility inputs
  for the Sound Life matching analysis.
- `06_CODE_RELEASE/current_native_sources.zip`: original model code, protocols
  and cell-allocation records. Model weights, prepared expression data and saved
  predictions are supplied in separate archives listed in the
  [upstream guide](../reconstruction/current_upstream/README.md).

For access, contact [Hao Zhou](mailto:hao.zhou@ku.ac.ae). The
[licence guide](../LICENSE_SCOPE.md) lists the terms for code and dataset-derived
files.

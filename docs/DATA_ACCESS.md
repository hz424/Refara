# Data access and redistribution

Numerical replays use bundled synthetic fixtures, pseudonymized arrays and score
tables. Further reconstruction uses the matching Source Data package or original
study inputs. [Licence scope](../LICENSE_SCOPE.md) identifies the applicable
source terms.

| Analysis | Available inputs and access |
|---|---|
| Norman genetic-combination prediction | Full reconstruction uses the original GSE133344 expression matrix. Bundled inputs include scored utilities, selection summaries and pair tables; the selected-CPA component also includes its checkpoint and fixed-panel arrays. Reviewer Source Data supplies normalized inputs, fitted models and gene-level predictions. Original dataset terms apply. |
| GSE162632 held-out analysis | [GEO GSE162632](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE162632) supplies the study data. Source Data includes pseudonymized utilities, cached predictions, control and treated means, gene identifiers and deterministic allocation indices. |
| GSE162632 39-donor follow-up | `Biological_validation_v1810/` in Source Data contains deidentified donor, batch and capture-orientation summaries. `evidence/organizer/transfer/` bundles reporting scores and aggregate membership checks. These donors were excluded from fitting and candidate screening but share the study and experimental batches with the discovery data. |
| Kang conditioning analysis | Four locally authored files in `Extended_Data_Figure_2_attribution/` provide eight-root summaries of shifts, five-endpoint diagnostic bounds, utility responses and standardized displacement. They exclude root correspondence, expression matrices and study-wide predictions. |
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

## Confidential review inputs

Two attachments are held separately for confidential review under the source
terms; they are not included in the public repository:

- `SOURCE_MEMBERSHIP_REVIEW_V189.zip`: original donor/cell correspondence,
  metadata and source projections for the Influenza reconstruction.
- `Sound_Life_Graph_Utilities_for_Review.zip`: anonymous graph, weights and
  witness inputs for Sound Life matching.

For these review attachments, contact [Hao Zhou](mailto:hao.zhou@ku.ac.ae).

GSE162632 summaries use `ROOT_01`–`ROOT_08`, linked to the source units through
controlled correspondence. Public code and aggregate Source Data exclude private
donor names, production cell identifiers and raw cell-expression matrices.
Sound Life summaries retain Allen terms and exclude individual records,
membership vectors and participant hashes. The 39-donor follow-up tests a fixed
observed-response contrast within the same study; it is not an external
replication of model selection.

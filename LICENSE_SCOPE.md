# Licence scope

- Python and shell code, tests and executable schemas: BSD-3-Clause
  ([`LICENSE`](LICENSE)).
- Locally authored documentation and the Figure 1 overview image
  (`docs/assets/figure-1.png`): CC BY 4.0
  ([`LICENSES/CC-BY-4.0.md`](LICENSES/CC-BY-4.0.md)). Separately distributed
  Source Data retain the file-specific terms recorded below.
- Font licence notices from the separately archived artwork are retained in
  [`LICENSES/archived_artwork_fonts.txt`](LICENSES/archived_artwork_fonts.txt);
  those artwork fonts are not included in this source package.
- The pseudonymized derived arrays in `capsules/gse162632_scgen/replay` are
  locally authored derivatives covered by CC BY 4.0. The separately packaged
  prepared scGen archive has the same licence.
- The pseudonymized label-level utility and allocation-interaction arrays in
  `capsules/gse162632_allocation` are locally authored GSE162632 derivatives
  covered by CC BY 4.0. They contain public root indices rather than source
  donor identifiers and do not extend this licence to the underlying GEO data.
- Third-party raw data, restricted records and study-wide production prediction
  arrays are not distributed or licensed by this repository. The selected Norman
  CPA checkpoint is included under the dataset-specific scope described below.

The deidentified summary tables under `analysis/frozen_model_evaluation/source_data`
are locally authored derivatives covered by CC BY 4.0. They contain only public
donor, batch and orientation aliases and do not extend this licence to the
underlying GEO expression data.

The separately distributed, locally authored GSE162632 root-utility,
within-construction contrast, paired-construction-change, multi-realization,
Cmodel-successor and manifest files are covered by the CC BY 4.0 grant. The root codes are
local pseudonyms; their mapping to study-specific donor–batch labels is not
distributed. This grant does not extend to underlying GEO expression data,
third-party model implementations or fitted checkpoints.

The separately distributed Parse-derived or Parse/GSE162632 mixed aggregate
tables are adapted from the Parse Biosciences 10-million-cell PBMC dataset.
They are excluded from the CC BY 4.0 grant and are available only for
noncommercial use under the source CC BY-NC 4.0 terms
([`LICENSES/CC-BY-NC-4.0.md`](LICENSES/CC-BY-NC-4.0.md)).

The current Figure 3 capsule includes compact Parse-derived or mixed numerical
summaries under `evidence/figure3/source/`, including `source/utility/parse/`.
These retain the same CC BY-NC 4.0 scope. Influenza PBMC-only summaries retain
their GSE162632 derivative scope above. The source manifest identifies each
file; packaging these summaries does not relicense the underlying data.
The Norman saved scores and small cached profiles under `evidence/table1/`
retain their source-specific terms and attribution, as described in that
companion. Author-written replay code is covered by BSD-3-Clause.

`evidence/organizer/data/` contains Norman GSE133344 task scores and control
geometry under those same dataset terms. Its source manifest
records the original score tables and their hashes. Cite Norman et al., Science
**365**, 786–793 (2019), doi:10.1126/science.aax4438 when reusing these derivatives.
Norman reporting-rule summaries in `evidence/organizer/expected/` retain this
scope. The pseudonymized GSE162632 scores and membership summaries in
`evidence/organizer/transfer/`, and their comparison summaries, are locally
authored derivatives under CC BY 4.0. That grant does not extend to the original
GEO data, third-party model implementations or fitted checkpoints.

The compact tables in `evidence/model_comparisons/` retain their dataset-specific
scope: GSE162632-only rows are CC BY 4.0; Parse-derived rows are CC BY-NC 4.0.
Their manifest identifies the sources. Numerical inputs in
`evidence/program_responses/` are locally authored GSE162632 derivatives under
CC BY 4.0 and contain program identifiers, not gene-set membership definitions.
The aggregate counts and matching-level utility sums in
`evidence/matching_sensitivity/` retain the Allen Institute noncommercial
research terms and attribution in the component's
[licence notice](evidence/matching_sensitivity/LICENSE_SCOPE.md). They are
excluded from the CC BY grant. Code in these components is BSD-3-Clause.

`evidence/gse181897_reference_design/` contains pseudonymized task-loss tables
from [GSE181897](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181897)
evaluated with fixed Parse-trained models. These model-derived tables and their
figures retain the Parse noncommercial scope above. The capsule records source
accessions and artifact hashes; it contains no raw expression, cell barcodes
or fitted checkpoints. Author-written analysis and plotting code is
BSD-3-Clause. This distribution does not relicense the underlying GEO data or
third-party model implementations.

The compressed Parse/GSE181897 loss arrays in
`evidence/reporting_reliability/`, the Parse-trained weights and small
derived count matrix in `examples/real_model_reference/`, and the exported
scGen weights in `examples/conditioned_model_reference/`, retain the Parse
CC BY-NC 4.0 scope. Their local provenance and licence notices identify the
inputs. These terms do not extend to unrelated code or grant additional rights
to the source datasets. Author-written code in these components is BSD-3-Clause.

Separately distributed Sound Life-derived aggregate display summaries are
provided for noncommercial research use subject to the Allen Institute Terms
of Use and are excluded from the CC BY 4.0 grant. Exact file lists and
attribution are in
[`data/derived/LICENSE_SCOPE.md`](data/derived/LICENSE_SCOPE.md).

The frozen losses and numerical summaries in `evidence/jerber_reference_audit/`
are Jerber-derived aggregate records and retain the source-specific provenance
and applicable terms recorded for the Jerber study in `docs/DATA_ACCESS.md`
and `data/derived/LICENSE_SCOPE.md`. They contain root indices, not source donor
identifiers. This addition grants no new rights to underlying expression data,
model weights or third-party software; none are bundled here. The replay code
is BSD-3-Clause. Cite Jerber et al. (2021), doi:10.1038/s41588-021-00801-6.

## Genetic-perturbation extension

`analysis/norman_reference/` and `analysis/norman_direct_effect/` contain author-written code, preparation or model-selection records and scored summaries derived from Norman et al., Science **365**, 786–793 (2019), GEO GSE133344. Their utility arrays store task-level scores and reference diagnostics; expression matrices, fitted coefficients and gene-level predictions use the separate reconstruction or review routes. Dataset derivatives retain the original source’s applicable terms and are not relicensed as analysis software. Cite the original study when reusing them. Each component has a `LICENSE_SCOPE.md` notice.

`analysis/norman_training_review/` supplies the current selected CPA checkpoint, fixed-panel effect arrays, scales, membership records and scored utilities for Supplementary Table 8 and Extended Data Figure 3. These are Norman dataset derivatives under the same source terms; see its `LICENSE_SCOPE.md`.

`analysis/low_replication/` contains author-generated simulations and numerical
reconstruction code. Notices for its separately archived artwork fonts remain
with that component.

## Public reference-audit example

`examples/vcc_audit/` contains author-written example and verification code under
the repository's BSD-3-Clause licence. Its default runner downloads the
Adamson et al. (2016) K562 10X005 dataset distributed by
scPerturb in [Zenodo record 13350497](https://zenodo.org/records/13350497)
under CC BY 4.0. Cite the original
[Adamson study](https://doi.org/10.1016/j.cell.2016.11.048) and the scPerturb
data resource when reusing its data. The optional Dixit K562 example is distributed
by the same record under CC BY 4.0; cite the original
[Dixit study](https://doi.org/10.1016/j.cell.2016.11.038) when reusing that resource.
The source matrices are downloaded separately;
the fixture records attribution, membership, transformations and provenance in
`SOURCE.json`.

The separately installed official
[cell-eval2 implementation](https://github.com/ArcInstitute/cell-eval2/tree/5e64833518a6603a0301cbe28185d49c30f4a986)
retains its MIT licence. Its optional H1 tutorial fixture retains its upstream
data terms. Neither third-party resource is relicensed as repository software.

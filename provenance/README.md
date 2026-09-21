# Source records

| Record | Contents |
|---|---|
| [SOURCE_DATA_ROOT_MANIFEST_V1.json](SOURCE_DATA_ROOT_MANIFEST_V1.json) | Current external Source Data files and checksums, including the corrected Jerber scGen results |
| [JERBER_SCGEN_CORRECTION.json](JERBER_SCGEN_CORRECTION.json) | Corrected and historical Jerber results, inference-code hashes and numerical checks |
| [V183_IMMUTABLE_INPUTS.json](V183_IMMUTABLE_INPUTS.json) | Original input hashes used by the correction checks; current replays use the manifest above |
| [PRIMARY_CONSTRUCTION_ALIAS_CROSSWALK.tsv](PRIMARY_CONSTRUCTION_ALIAS_CROSSWALK.tsv) | Reference-construction names used in the paper and analysis files |
| [README_FIGURE.json](README_FIGURE.json) | Source and rendering details for the homepage illustration |

Individual replay datasets include their own manifests. The two Jerber files
whose names end in `V1` retain those names for compatibility and contain the
corrected values. The correction applies each cell-type shift to every donor;
its full numerical comparison is in the correction record above.

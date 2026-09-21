# External Source Data

Obtain the separate Source Data package and pass its extracted `Source_Data`
directory with `--source-data-root`. Submission packages call the same directory
`05_SOURCE_DATA`. The [reproduction guide](../docs/ADDITIONAL_ANALYSES.md#aggregate-replay-with-source-data)
gives the environment and aggregate-replay command.

The replay checks files against
[`provenance/SOURCE_DATA_ROOT_MANIFEST_V1.json`](../provenance/SOURCE_DATA_ROOT_MANIFEST_V1.json)
and copies them to temporary paths, leaving your Source Data unchanged.
Version identifiers in the manifest identify the required data component.

This repository's `data/derived/` contains licence documentation and the generic
inference-qualification report. The two bundled GSE162632 replay datasets are
under `capsules/`. See [data access](../docs/DATA_ACCESS.md) and
[file-specific terms](derived/LICENSE_SCOPE.md).

To check the additional metric summaries for the original Figure 3, run
`analysis/metric_summaries/validate.py` as shown in the
[metric-summary guide](../analysis/metric_summaries/README.md).

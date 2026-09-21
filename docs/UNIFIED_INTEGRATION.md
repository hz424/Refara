# Runtime and validation notes

The toolkit uses the `reference_design` Python module and `reference-design`
command. See the [scoring guide](REFERENCE_DESIGN.md) for MSE plans,
[organizer guide](organizer.md) for comparison reporting and
[VCC guide](VCC_AUDIT.md) for official raw-cell metrics.

## Environments

The standard Python 3.10 installation combines `vcc-py310-linux.txt` and
`unified-prepare-py310.txt`. The [VCC installer](VCC_INSTALLATION.md) also creates
a separate Python 3.12 environment for the official scorer, `cell-eval2 0.16.0`
at commit `5e64833518a6603a0301cbe28185d49c30f4a986`.

`locked.txt` and `prepare-locked.txt` retain the Python 3.10.17 / NumPy 1.26.4
environment for the [additional recorded analyses](ADDITIONAL_ANALYSES.md).
Use a separate environment for those commands.

<a id="inherited-executed-evidence"></a>
<a id="historical-vcc-checks"></a>

## Official-scorer validation

The public Adamson example was checked with the 1.8.20.dev1 adapter and the
pinned official scorer above. Three example predictions across nine allocations
produced 27 outcomes: 15 scored model–allocation combinations and 12 unavailable
because calibration failed, with no failed or missing workers.
All 189 metric rows and 189 model-pair rows were retained. Four prespecified
shared-depth allocations failed the official baseline-scale calibration;
the completed audit returned exit code 2.

Independent calculations covered the original and first separated allocations:
248 numerical and four structural-null comparisons passed, with maximum absolute
difference `1.1102230246251565e-16` at absolute and relative tolerances of `1e-10`.
They also reproduced the calibration rejection for the first shared allocation.
The remaining unavailable allocations were checked for matching input hashes
and complete unavailable-result rows.

These checks use fixed example predictions to test agreement with the official
scorer. Predictive performance on held-out VCC contexts requires separate
evaluation. VCC's normalized expression-error metric uses a ratio of panel sums;
the MSE protocol averages task scores with its declared unit weights. Organizer
reporting reads supplied score and reference-distance tables. Converting VCC
results into those tables currently requires a separate preparation step.

The [VCC installation guide](VCC_INSTALLATION.md#inherited-vcc-validation)
identifies the recorded example and current environment checks. The current CI
runs the default tests, official-worker tests, examples and wheel checks.

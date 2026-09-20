# Complete-family structural-zero correction

This correction sets ten algebraically zero direct-effect rotation-minus-Shared paired contrasts to exact zero before studentization. It retains all 56 contrast identities and all 448 root rows. Original floating-point values and their historical diagnostics stay in the legacy Source Data component.

For fixed direct effects a and b and observation target z, the squared-loss utility contrast is a weighted affine function of z: U(a,z)-U(b,z) = 2<a-b,z> + ||b||²-||a||². The equal-depth rotation averages z to the Shared-union observation target. Consequently its direct/direct contrast has exactly the same mean as Shared at each root-task stratum, and the rotation-minus-Shared change is structurally zero before root aggregation. The identity requires fixed model outputs, squared loss, equal block depth and scoring before averaging; it holds regardless of observed effect size.

For these ten known members, the root change, mean, standard error, bounds and every multiplier statistic are defined as zero. Other members use the original centred, re-studentized exhaustive-sign diagnostic. Any invalid standard error in a nonstructural member makes the entire family unbounded. All 56 members remain in the maximum. The corrected critical value is 4.729828863130656; 33 nondegenerate zero-excluding flags and the focal reversals are unchanged. The 84-member within-construction family is unchanged. The corrected bounds still lack calibrated 95% coverage.

```bash
python analysis/diagnostic_correction_v189/replay.py --source-data-root SOURCE_DATA --verify
# Recompute into a separate review directory:
python analysis/diagnostic_correction_v189/replay.py --source-data-root SOURCE_DATA --output-dir new-correction
```

The verifier checks all 448 keys, 56 identities, the design-based mask, recomputed source bytes, alternate algebraic order, lossless decimal float64 round trip, structural-noise invariance, and unchanged nondegenerate flags. The corrected summary is plotted only after this verification.

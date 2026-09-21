#!/usr/bin/env python3
"""Dependency-free numerical primitives for V3 comparator V1.

The comparator uses a one-sample Gaussian-root t procedure.  Python's standard
library has no Student-t CDF, so this module freezes a regularized incomplete
beta implementation rather than importing an unbound site-package.  The
successor degrees of freedom are 1, 2, 3, 5, 7, 11, 19, and 39.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Mapping, Sequence


BETA_CF_MAX_ITERATIONS = 10_000
BETA_CF_RELATIVE_TOLERANCE = 2.0 ** -48
BETA_CF_FPMIN = sys.float_info.min / BETA_CF_RELATIVE_TOLERANCE
SUPPORTED_T_DF = frozenset({1, 2, 3, 5, 7, 11, 19, 39})


class ComparatorNumericalError(RuntimeError):
    """Raised when a frozen comparator numerical contract is violated."""


@dataclass(frozen=True)
class OneSampleTResult:
    mean: float
    sample_variance: float
    standard_error: float
    statistic: float
    one_sided_p_value: float
    zero_standard_error_fail_closed: bool


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparatorNumericalError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ComparatorNumericalError(f"{name} must be finite")
    return result


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Modified-Lentz continued fraction for incomplete beta."""

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < BETA_CF_FPMIN:
        d = BETA_CF_FPMIN
    d = 1.0 / d
    value = d

    for iteration in range(1, BETA_CF_MAX_ITERATIONS + 1):
        twice = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < BETA_CF_FPMIN:
            d = BETA_CF_FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < BETA_CF_FPMIN:
            c = BETA_CF_FPMIN
        d = 1.0 / d
        value *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + coefficient * d
        if abs(d) < BETA_CF_FPMIN:
            d = BETA_CF_FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < BETA_CF_FPMIN:
            c = BETA_CF_FPMIN
        d = 1.0 / d
        multiplier = d * c
        value *= multiplier
        if abs(multiplier - 1.0) <= BETA_CF_RELATIVE_TOLERANCE:
            if not math.isfinite(value) or value <= 0.0:
                raise ComparatorNumericalError(
                    "incomplete-beta continued fraction returned invalid value"
                )
            return value

    raise ComparatorNumericalError(
        "incomplete-beta continued fraction did not converge"
    )


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return I_x(a,b) for finite positive a,b and x in [0,1]."""

    x = _require_finite_number(x, "x")
    a = _require_finite_number(a, "a")
    b = _require_finite_number(b, "b")
    if not 0.0 <= x <= 1.0:
        raise ComparatorNumericalError("x must be in [0,1]")
    if a <= 0.0 or b <= 0.0:
        raise ComparatorNumericalError("a and b must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        result = front * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - (
            front * _beta_continued_fraction(b, a, 1.0 - x) / b
        )
    if not math.isfinite(result):
        raise ComparatorNumericalError("regularized incomplete beta is non-finite")
    return min(1.0, max(0.0, result))


def student_t_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Return P(T_df >= statistic) for the frozen comparator degrees of freedom."""

    if isinstance(degrees_of_freedom, bool) or not isinstance(
        degrees_of_freedom, int
    ):
        raise ComparatorNumericalError("degrees_of_freedom must be an integer")
    if degrees_of_freedom not in SUPPORTED_T_DF:
        raise ComparatorNumericalError(
            f"unsupported degrees_of_freedom: {degrees_of_freedom}"
        )
    if isinstance(statistic, bool) or not isinstance(statistic, (int, float)):
        raise ComparatorNumericalError("statistic must be a real number")
    statistic = float(statistic)
    if math.isnan(statistic):
        raise ComparatorNumericalError("statistic must not be NaN")
    if statistic == math.inf:
        return 0.0
    if statistic == -math.inf:
        return 1.0
    if statistic == 0.0:
        return 0.5

    df = float(degrees_of_freedom)
    x = df / (df + statistic * statistic)
    half_beta = 0.5 * regularized_incomplete_beta(x, 0.5 * df, 0.5)
    result = half_beta if statistic > 0.0 else 1.0 - half_beta
    return min(1.0, max(0.0, result))


def one_sample_root_t_positive(contrasts: Sequence[float]) -> OneSampleTResult:
    """One-sided test of positive mean for one contrast per independent root.

    The sample variance uses denominator R-1 and SE=s/sqrt(R).  This is the
    intercept-only root-row CR1 calculation.  Exact zero SE fails closed with
    p=1 and is separately flagged.
    """

    values = tuple(
        _require_finite_number(value, f"contrasts[{index}]")
        for index, value in enumerate(contrasts)
    )
    count = len(values)
    if count - 1 not in SUPPORTED_T_DF:
        raise ComparatorNumericalError(
            f"root count {count} is outside the frozen comparator grid"
        )
    mean = math.fsum(values) / count
    squared_error = math.fsum((value - mean) ** 2 for value in values)
    sample_variance = squared_error / (count - 1)
    standard_error = math.sqrt(sample_variance / count)
    if standard_error == 0.0:
        return OneSampleTResult(
            mean=mean,
            sample_variance=sample_variance,
            standard_error=0.0,
            statistic=0.0,
            one_sided_p_value=1.0,
            zero_standard_error_fail_closed=True,
        )
    statistic = mean / standard_error
    return OneSampleTResult(
        mean=mean,
        sample_variance=sample_variance,
        standard_error=standard_error,
        statistic=statistic,
        one_sided_p_value=student_t_survival(statistic, count - 1),
        zero_standard_error_fail_closed=False,
    )


def holm_common_denominator(
    p_value_numerators: Mapping[str, int],
    p_value_denominator: int,
    *,
    alpha_numerator: int = 1,
    alpha_denominator: int = 20,
) -> tuple[str, ...]:
    """Exact Holm decisions for non-negative p numerators with one denominator."""

    if (
        isinstance(p_value_denominator, bool)
        or not isinstance(p_value_denominator, int)
        or p_value_denominator <= 0
    ):
        raise ComparatorNumericalError("p_value_denominator must be positive")
    if not p_value_numerators:
        raise ComparatorNumericalError("p-value family must not be empty")
    checked: list[tuple[int, str]] = []
    for hypothesis_id, numerator in p_value_numerators.items():
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ComparatorNumericalError("hypothesis IDs must be non-empty strings")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or not 0 <= numerator <= p_value_denominator
        ):
            raise ComparatorNumericalError("invalid p-value numerator")
        checked.append((numerator, hypothesis_id))

    ordered = sorted(checked, key=lambda item: (item[0], item[1]))
    family_size = len(ordered)
    rejected: list[str] = []
    for index, (numerator, hypothesis_id) in enumerate(ordered):
        remaining = family_size - index
        if (
            numerator * alpha_denominator * remaining
            <= alpha_numerator * p_value_denominator
        ):
            rejected.append(hypothesis_id)
        else:
            break
    return tuple(rejected)


def holm_float64(
    p_values: Mapping[str, float],
    *,
    alpha_numerator: int = 1,
    alpha_denominator: int = 20,
) -> tuple[str, ...]:
    """Deterministic Holm decisions for the CR1 Student-t p-values."""

    if not p_values:
        raise ComparatorNumericalError("p-value family must not be empty")
    checked: list[tuple[float, str]] = []
    for hypothesis_id, p_value in p_values.items():
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ComparatorNumericalError("hypothesis IDs must be non-empty strings")
        p_value = _require_finite_number(p_value, f"p_values[{hypothesis_id!r}]")
        if not 0.0 <= p_value <= 1.0:
            raise ComparatorNumericalError("p-values must be in [0,1]")
        checked.append((p_value, hypothesis_id))

    ordered = sorted(checked, key=lambda item: (item[0], item[1]))
    family_size = len(ordered)
    rejected: list[str] = []
    for index, (p_value, hypothesis_id) in enumerate(ordered):
        remaining = family_size - index
        threshold = alpha_numerator / (alpha_denominator * remaining)
        if p_value <= threshold:
            rejected.append(hypothesis_id)
        else:
            break
    return tuple(rejected)

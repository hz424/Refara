"""Classify supplied paired-loss intervals against declared reporting directions.

Margins always mean loss(model_b) - loss(model_a): a positive margin favours A.
This module neither estimates intervals nor verifies their construction,
multiplicity correction, experimental independence or prospective declaration.
Its output is a mechanical summary of the supplied records, not a certificate
that a reporting rule is reliable. Candidate comparisons and reporting decisions
should be frozen before examining their validation intervals.
"""
from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Literal


Direction = Literal["a", "b", "hold"]


def _label(value: str, name: str) -> None:
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or any(character in value for character in "\t\r\n")):
        raise ValueError(f"{name} must be a nonempty identifier without surrounding whitespace or line breaks")


@dataclass(frozen=True)
class Comparison:
    """One declared task and unordered model pair, with a fixed A/B orientation.

    A task identifier may encode cell type, perturbation and dose/time; its
    biological meaning is supplied by the caller and is not inferred here.
    """

    task: str
    model_a: str
    model_b: str

    def __post_init__(self) -> None:
        for name in ("task", "model_a", "model_b"):
            _label(getattr(self, name), name)
        if self.model_a == self.model_b:
            raise ValueError("A comparison requires two distinct models")


@dataclass(frozen=True)
class ReportedDirection:
    """A published A/B direction or explicit hold, plus its external interval.

    ``interval`` is always on the loss(B) - loss(A) scale, including for a B
    direction. Bounds must be finite and ordered. A hold can omit its interval;
    a supplied hold interval is checked but never assigned a direction.
    Bounds are stored as Python floats. Raw ordering is checked before conversion;
    overflow and nonzero bounds that underflow to zero are rejected so conversion
    cannot change which side of zero an interval occupies.
    """

    comparison: Comparison
    direction: Direction
    interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, Comparison):
            raise ValueError("Report comparison must be a Comparison")
        if not isinstance(self.direction, str) or self.direction not in ("a", "b", "hold"):
            raise ValueError("Direction must be 'a', 'b' or 'hold'")
        if self.interval is None:
            if self.direction != "hold":
                raise ValueError("A published direction requires an interval")
            return
        if isinstance(self.interval, (str, bytes, bytearray, Mapping, Set)):
            raise ValueError("Interval must contain exactly two finite real bounds")
        try:
            bounds = tuple(self.interval)
        except TypeError as exc:
            raise ValueError("Interval must contain exactly two finite real bounds") from exc
        if len(bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, Real) for v in bounds):
            raise ValueError("Interval must contain exactly two finite real bounds")
        if bounds[0] > bounds[1]:
            raise ValueError("Interval lower bound exceeds its upper bound")
        try:
            lower, upper = (float(value) for value in bounds)
        except (OverflowError, ValueError) as exc:
            raise ValueError("Interval bounds must be finite") from exc
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Interval bounds must be finite")
        if any(raw != 0 and converted == 0 for raw, converted in zip(bounds, (lower, upper))):
            raise ValueError("Nonzero interval bound underflows to zero as a Python float")
        object.__setattr__(self, "interval", (lower, upper))


def summarize_directions(roster: Iterable[Comparison],
                         reports: Iterable[ReportedDirection]) -> dict:
    """Return counts, coverage and reversal/unresolved fractions for one rule.

    Every candidate in the nonempty roster must have exactly one record,
    including explicit holds. Reports may arrive in any order; output follows
    roster order. Reversing A/B in a duplicate pair cannot create a new candidate.

    For a published direction, orient its interval toward the favoured model.
    A strictly positive interval is supported, a strictly negative interval is
    reversed, and an interval touching or crossing zero is unresolved. No tie
    tolerance or direction is inferred from a point estimate.

    N is the fixed candidate count and M is the number of published directions.
    Coverage is M/N. Reversal and unresolved fractions use the same M, including
    unresolved publications. Both fractions are None when every candidate is held.
    The module does not match coverage between different reporting rules.
    """
    try:
        candidates, records = tuple(roster), tuple(reports)
    except TypeError as exc:
        raise ValueError("Roster and reports must be iterable") from exc
    if not candidates or any(not isinstance(item, Comparison) for item in candidates):
        raise ValueError("Roster must contain at least one Comparison")
    identities = [(item.task, frozenset((item.model_a, item.model_b))) for item in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("Roster contains a duplicate task/model pair, possibly with reversed model order")
    declared = set(candidates)
    lookup = {}
    for record in records:
        if not isinstance(record, ReportedDirection):
            raise ValueError("Reports must contain ReportedDirection records")
        if record.comparison not in declared:
            raise ValueError("Report comparison or model orientation is not in the roster")
        if record.comparison in lookup:
            raise ValueError("Duplicate report for a declared comparison")
        lookup[record.comparison] = record
    if set(lookup) != declared:
        raise ValueError("Every roster comparison requires an explicit direction or hold record")

    counts = {"candidates": len(candidates), "published": 0, "supported": 0,
              "reversed": 0, "unresolved": 0, "held": 0}
    comparisons = []
    for candidate in candidates:
        record = lookup[candidate]
        oriented = None
        if record.direction == "hold":
            status = "hold"
            counts["held"] += 1
        else:
            lower, upper = record.interval
            oriented = (lower, upper) if record.direction == "a" else (-upper, -lower)
            status = "supported" if oriented[0] > 0 else "reversed" if oriented[1] < 0 else "unresolved"
            counts["published"] += 1
            counts[status] += 1
        comparisons.append({"task": candidate.task, "model_a": candidate.model_a,
                            "model_b": candidate.model_b, "direction": record.direction,
                            "interval": None if record.interval is None else list(record.interval),
                            "oriented_interval": None if oriented is None else list(oriented),
                            "status": status})
    published = counts["published"]
    return {
        "schema_version": 1,
        "margin_definition": "loss(model_b) - loss(model_a)",
        "counts": counts,
        "coverage": published / len(candidates),
        "reversal_fraction": counts["reversed"] / published if published else None,
        "unresolved_fraction": counts["unresolved"] / published if published else None,
        "comparisons": comparisons,
        "interpretation": "Classification of provided intervals only; interval construction, multiplicity and biological independence are not verified.",
    }

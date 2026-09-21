"""Candidate independent-unit intervals and finite-panel directional-risk bounds.

This pure in-memory module does not read data, verify experimental independence,
or establish coverage calibration. Inputs are already averaged paired losses,
one row per declared independent native block and one column per frozen
comparison. Cells, model seeds and reference allocations are not additional rows.
The present public entry deliberately accepts synthetic engineering mode only.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from typing import Mapping

import numpy as np
from scipy.stats import t as student_t
import scipy

MODE = "synthetic_engineering_only"
METHOD = "bonferroni_paired_t_joint_max_studentized_bootstrap_envelope_candidate_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _labels(values, name):
    _require(not isinstance(values, (str, bytes)), name + " must be a sequence")
    values = list(values)
    _require(values and all(isinstance(x, str) and x.strip() == x and x for x in values),
             name + " needs nonempty string labels")
    _require(len(set(values)) == len(values), "Duplicate " + name)
    return values


def _integer(value, name, minimum):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
             and value >= minimum, "Invalid " + name)
    return int(value)


def _real_array(value, name, allow_missing=False):
    raw = np.asarray(value)
    _require(raw.dtype.kind in "fiu", name + " must contain real numbers")
    result = np.asarray(raw, dtype=np.float64)
    _require(not np.isinf(result).any(), name + " contains infinite or overflowing values")
    if raw.dtype.kind in "iu":
        _require(all(int(before) == int(after) for before, after in zip(raw.flat, result.flat)),
                 name + " loses integer precision in float64 conversion")
    _require(not np.any((raw != 0) & (result == 0)), name + " underflows a nonzero value to zero")
    if not allow_missing:
        _require(np.isfinite(result).all(), name + " contains missing values")
    return result


def _digest_array(value):
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([arr.dtype.str, list(arr.shape)], separators=(",", ":")).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _sha_label(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _verify_structural_zero(evidence, comparison, unit_ids, column):
    """Check an explicit identity contract plus its supplied numerical evidence.

    The caller must authenticate the predictor/score/target digests against its
    frozen manifests. Equal observed predictions alone cannot establish that
    two predictors are the same function on future experimental units.
    """
    fields = {"basis", "comparison_id", "unit_ids", "predictor_function_sha256_a",
              "predictor_function_sha256_b", "prediction_contract_sha256_a",
              "prediction_contract_sha256_b", "score_contract_sha256_a",
              "score_contract_sha256_b", "target_sha256_a", "target_sha256_b",
              "predictions_a", "predictions_b", "loss_a", "loss_b"}
    _require(isinstance(evidence, Mapping) and set(evidence) == fields,
             "Structural-zero evidence schema differs")
    _require(evidence["basis"] == "identical_frozen_predictor_and_scoring_contract"
             and evidence["comparison_id"] == comparison, "Structural-zero identity declaration differs")
    supplied_ids = _labels(evidence["unit_ids"], "structural-zero units")
    _require(set(supplied_ids) == set(unit_ids), "Structural-zero unit support differs")
    order = [supplied_ids.index(unit) for unit in unit_ids]
    receipt = {"basis": evidence["basis"], "comparison_id": comparison, "unit_ids": unit_ids}
    for stem in ["predictor_function", "prediction_contract", "score_contract"]:
        a, b = (evidence[stem + "_sha256_" + suffix] for suffix in ["a", "b"])
        _require(_sha_label(a) and a == b, "Structural-zero " + stem + " identity differs")
        receipt[stem + "_sha256"] = a
    targets = []
    for suffix in ["a", "b"]:
        values = list(evidence["target_sha256_" + suffix])
        _require(len(values) == len(unit_ids) and all(_sha_label(v) for v in values),
                 "Structural-zero target identity missing")
        targets.append([values[j] for j in order])
    _require(targets[0] == targets[1], "Structural-zero paired targets differ")
    receipt["target_sha256"] = targets[0]
    pa, pb = (np.asarray(evidence["predictions_" + suffix]) for suffix in ["a", "b"])
    for arr in [pa, pb]:
        _real_array(arr, "Structural-zero prediction")
        _require(arr.ndim >= 2 and arr.shape[0] == len(unit_ids) and all(arr.shape),
                 "Structural-zero prediction axes differ")
    _require(pa.shape == pb.shape and pa.dtype == pb.dtype, "Structural-zero prediction representations differ")
    pa, pb = pa[order], pb[order]
    _require(_digest_array(pa) == _digest_array(pb), "Structural-zero predictions are not byte-identical")
    raw_la, raw_lb = (np.asarray(evidence["loss_" + suffix]) for suffix in ["a", "b"])
    # Check the supplied representation before a float64 conversion can merge
    # distinct integer values (for example 2**53 and 2**53+1).
    _require(raw_la.shape == raw_lb.shape and raw_la.dtype == raw_lb.dtype
             and _digest_array(raw_la) == _digest_array(raw_lb),
             "Structural-zero raw paired losses differ")
    la, lb = (_real_array(arr, "Structural-zero paired loss") for arr in [raw_la, raw_lb])
    _require(la.shape == lb.shape == (len(unit_ids),) and np.all(la >= 0) and np.all(lb >= 0),
             "Structural-zero loss axes or signs differ")
    la, lb = la[order], lb[order]
    _require(np.array_equal(la, lb) and np.isfinite(column).all() and np.all(column == 0),
             "Structural-zero paired losses do not agree with the input matrix")
    receipt.update(predictions_sha256=_digest_array(pa), paired_loss_sha256=_digest_array(la),
                   caller_must_authenticate_frozen_contract_hashes=True)
    return receipt


def _mean_and_se(values, axis):
    """Stable constant handling and scaled standard error; no unit weights."""
    n = values.shape[axis]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Mean by an offset preserves exact constants; extreme overflow remains
        # an explicit unassessable numerical failure rather than a narrow CI.
        origin = np.take(values, 0, axis=axis)
        mean = origin + np.mean(values - np.expand_dims(origin, axis), axis=axis)
        centered = values - np.expand_dims(mean, axis)
        scale = np.max(np.abs(centered), axis=axis)
        normalized = np.divide(centered, np.expand_dims(scale, axis),
                               out=np.zeros_like(centered), where=np.expand_dims(scale > 0, axis))
        se = scale * np.sqrt(np.sum(normalized * normalized, axis=axis) / (n * (n - 1)))
    return mean, se


def _extended_studentized(numerator, denominator):
    _require(np.isfinite(numerator).all() and np.isfinite(denominator).all()
             and np.all(denominator >= 0), "Studentization input is nonfinite")
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    nonzero_over_zero = (denominator == 0) & (numerator != 0)
    result[nonzero_over_zero] = np.sign(numerator[nonzero_over_zero]) * np.inf
    return result


def paired_matrix_intervals(margins, *, unit_ids, comparison_ids,
                            mode, confidence=0.95, resamples=19999, seed=20260923,
                            minimum_units=24, structural_zero_evidence=None,
                            bootstrap_indices=None):
    """Construct the draft interval envelope from an equal-unit paired matrix.

    NaN means a missing margin and makes that entire comparison unassessable;
    there is no available-case deletion. Infinity is rejected. Structural zeros
    require the explicit, checked identity-evidence schema above. A column that
    merely happens to be constant (even constant zero) receives no exact CI.

    ``bootstrap_indices`` is an optional explicit [resamples, n_units] fixture,
    indexed in lexicographically sorted unit order, enabling exact small golden
    checks. Generated resamples use the same canonical order and are shared
    across every comparison. No filesystem or real-input loader is provided.
    """
    _require(mode == MODE, "Candidate intervals accept synthetic engineering mode only")
    ids = _labels(unit_ids, "unit IDs")
    comparisons = _labels(comparison_ids, "comparison IDs")
    x = _real_array(margins, "Paired unit matrix", allow_missing=True)
    _require(x.ndim == 2 and x.shape == (len(ids), len(comparisons)),
             "Expected matrix [n_native_units, N_comparisons]")
    _require(len(ids) >= 2, "At least two declared independent units are required")
    _require(isinstance(confidence, Real) and not isinstance(confidence, (bool, np.bool_))
             and math.isfinite(confidence) and 0 < confidence < 1, "Invalid confidence")
    resamples = _integer(resamples, "bootstrap count", 2)
    seed = _integer(seed, "bootstrap seed", 0)
    minimum_units = _integer(minimum_units, "minimum independent units", 2)
    order = np.argsort(ids)
    ids = [ids[j] for j in order]
    x = x[order]
    n, N = x.shape
    proofs = {} if structural_zero_evidence is None else structural_zero_evidence
    _require(isinstance(proofs, Mapping) and set(proofs) <= set(comparisons), "Unknown structural-zero declaration")
    if bootstrap_indices is not None:
        indices = np.asarray(bootstrap_indices)
        _require(indices.dtype.kind in "iu" and indices.shape == (resamples, n),
                 "Explicit bootstrap-index axes differ")
        _require(np.all(indices >= 0) and np.all(indices < n), "Bootstrap unit index out of bounds")
    else:
        indices = np.random.default_rng(seed).integers(0, n, size=(resamples, n))

    alpha = 1 - float(confidence)
    critical_t = float(student_t.isf(alpha / (2 * N), n - 1))
    records, active = {}, []
    means, errors = np.full(N, np.nan), np.full(N, np.nan)
    for j, identity in enumerate(comparisons):
        column = x[:, j]
        record = dict(comparison_id=identity, independent_units=n,
                      observed_units=int(np.isfinite(column).sum()), mean=None, standard_error=None,
                      status="unassessable", interval=None, t_interval=None, bootstrap_interval=None,
                      reason=None, structural_zero_evidence=None)
        records[identity] = record
        if identity in proofs:
            record["structural_zero_evidence"] = _verify_structural_zero(proofs[identity], identity, ids, column)
            record.update(status="structural_zero", mean=0.0, standard_error=0.0,
                          interval=[0.0, 0.0], t_interval=[0.0, 0.0], bootstrap_interval=[0.0, 0.0],
                          reason="identical_frozen_predictor_and_scoring_contract")
            continue
        if not np.isfinite(column).all():
            record.update(reason="missing_unit_margin", missing_unit_ids=[ids[k] for k in np.flatnonzero(~np.isfinite(column))])
            continue
        mean, se = _mean_and_se(column, axis=0)
        means[j], errors[j] = float(mean), float(se)
        if np.isfinite(mean):
            record["mean"] = float(mean)
        if np.isfinite(se):
            record["standard_error"] = float(se)
        if n < minimum_units:
            record["reason"] = "insufficient_independent_units"
        elif not np.isfinite(mean) or not np.isfinite(se):
            record["reason"] = "nonfinite_original_sampling_statistic"
        elif se == 0:
            record["reason"] = "zero_original_SE_without_structural_identity"
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                low, high = mean - critical_t * se, mean + critical_t * se
            if not np.isfinite([low, high]).all():
                record["reason"] = "nonfinite_t_interval"
            else:
                record["t_interval"] = [float(low), float(high)]
                active.append(j)

    maxima = np.empty(resamples, dtype=float)
    zero_over_zero = nonzero_over_zero = 0
    bootstrap_status = "not_computed_all_structural_or_unassessable"
    critical_bootstrap = None
    maxima_receipt = None
    if active:
        matrix = x[:, active]
        # The same unit indices are used for every column, preserving task/pair
        # dependence. Chunking affects memory use only, not the random draws.
        chunk = min(128, max(1, (64 << 20) // (8 * n * len(active))))
        failed = False
        for start in range(0, resamples, chunk):
            stop = min(resamples, start + chunk)
            boot_mean, boot_se = _mean_and_se(matrix[indices[start:stop]], axis=1)
            numerator = boot_mean - means[active]
            if not np.isfinite(numerator).all() or not np.isfinite(boot_se).all():
                failed = True
                break
            zero_over_zero += int(np.count_nonzero((boot_se == 0) & (numerator == 0)))
            nonzero_over_zero += int(np.count_nonzero((boot_se == 0) & (numerator != 0)))
            values = _extended_studentized(numerator, boot_se)
            maxima[start:stop] = np.max(np.abs(values), axis=1)
        if failed:
            bootstrap_status = "unavailable_nonfinite_resample_statistic"
        else:
            q_index = math.ceil(float(confidence) * (resamples - 1))
            critical = float(np.sort(maxima)[q_index])
            maxima_receipt = dict(sha256=_digest_array(maxima),
                                  positive_infinite_draws=int(np.isposinf(maxima).sum()),
                                  zero_draws=int((maxima == 0).sum()), quantile_index=q_index)
            if math.isfinite(critical):
                critical_bootstrap = critical
                bootstrap_status = "finite"
            else:
                bootstrap_status = "unbounded_positive_infinite_critical_quantile"
        for j in active:
            record = records[comparisons[j]]
            if critical_bootstrap is None:
                record["reason"] = bootstrap_status
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                low = means[j] - critical_bootstrap * errors[j]
                high = means[j] + critical_bootstrap * errors[j]
            if not np.isfinite([low, high]).all():
                record["reason"] = "nonfinite_bootstrap_interval"
                continue
            record["bootstrap_interval"] = [float(low), float(high)]
            record.update(status="assessable", reason=None,
                          interval=[min(record["t_interval"][0], float(low)),
                                    max(record["t_interval"][1], float(high))])

    return dict(schema_version=1, mode=MODE, method=METHOD,
                implementation_status="candidate_synthetic_checks_only",
                coverage_calibration_status="not_run_pending_frozen_simulation_manifest",
                weighting="equal_independent_native_units", unit_ids=ids, comparison_ids=comparisons,
                independent_unit_count=n, multiplicity_N=N, confidence=float(confidence),
                degrees_of_freedom=n - 1, minimum_units=minimum_units,
                t_critical=critical_t, bootstrap_critical=critical_bootstrap,
                bootstrap_status=bootstrap_status, bootstrap_resamples=resamples, bootstrap_seed=seed,
                bootstrap_indices_source="provided_fixture" if bootstrap_indices is not None else "numpy_default_rng",
                bootstrap_indices_sha256=_digest_array(indices),
                bootstrap_maxima=maxima_receipt,
                extended_studentization_counts=dict(zero_over_zero=zero_over_zero, nonzero_over_zero=nonzero_over_zero),
                matrix_sha256=_digest_array(x), records=records,
                runtime=dict(numpy=np.__version__, scipy=scipy.__version__),
                scope="Approximate paired-mean simultaneous intervals conditional on fixed models, publication decisions and roster. Independence is supplied by the study design, not inferred from matrix rows. No population confidence interval over model pairs is implied.")


def _interval_record(record):
    _require(isinstance(record, Mapping), "Assessment record must be a mapping")
    _require(record.get("status") in ("assessable", "structural_zero", "unassessable"), "Unknown assessment status")
    interval = record.get("interval")
    if record["status"] == "unassessable":
        _require(interval is None and bool(record.get("reason")), "Unassessable record needs null interval and reason")
        return None
    _require(not isinstance(interval, (str, bytes, Mapping)) and interval is not None,
             "Assessable record needs an interval")
    bounds = tuple(interval)
    _require(len(bounds) == 2 and all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_)) for v in bounds),
             "Interval needs two real bounds")
    _require(bounds[0] <= bounds[1], "Interval bounds are reversed before conversion")
    lo, hi = map(float, bounds)
    _require(math.isfinite(lo) and math.isfinite(hi), "Interval must be finite")
    _require(not any(raw != 0 and converted == 0 for raw, converted in zip(bounds, [lo, hi])),
             "Interval bound underflows to zero")
    if record["status"] == "structural_zero":
        _require(lo == hi == 0 and record.get("structural_zero_evidence"), "Structural-zero interval needs identity evidence")
    return lo, hi


def possible_signs(record):
    """Possible population mean signs under the common supplied interval."""
    bounds = _interval_record(record)
    if bounds is None:
        return (-1, 0, 1)
    lo, hi = bounds
    return tuple(s for s, allowed in [(-1, lo < 0), (0, lo <= 0 <= hi), (1, hi > 0)] if allowed)


def _panel_inputs(comparison_ids, directions, records):
    identities = _labels(comparison_ids, "comparison IDs")
    _require(isinstance(directions, Mapping) and set(directions) == set(identities), "Every candidate needs an explicit publication direction or hold")
    _require(isinstance(records, Mapping) and set(records) == set(identities), "Assessment must cover the complete frozen roster")
    _require(all(v in ("a", "b", "hold") for v in directions.values()), "Unknown published direction")
    for identity in identities:
        _require(records[identity].get("comparison_id", identity) == identity, "Assessment comparison identity differs")
        _interval_record(records[identity])
    return identities


def finite_panel_risk_bounds(comparison_ids, directions, records):
    """Classify all publications and bound true strict reversal risk on this panel.

    Bounds inherit the common simultaneous interval assumptions. They are not
    binomial intervals, and do not treat model pairs/tasks as independent units.
    """
    ids = _panel_inputs(comparison_ids, directions, records)
    counts = dict(candidates=len(ids), published=0, held=0, supported=0, reversed=0, unresolved=0,
                  unassessable_publications=0)
    minimum = maximum = 0
    details = []
    for identity in ids:
        direction = directions[identity]
        signs = possible_signs(records[identity])
        if direction == "hold":
            counts["held"] += 1
            details.append(dict(comparison_id=identity, direction=direction, status="hold", possible_signs=list(signs)))
            continue
        counts["published"] += 1
        d = 1 if direction == "a" else -1
        bounds = _interval_record(records[identity])
        if bounds is None:
            classification, oriented = "unresolved", None
            counts["unassessable_publications"] += 1
        else:
            lo, hi = bounds
            oriented = [lo, hi] if d == 1 else [-hi, -lo]
            classification = "supported" if oriented[0] > 0 else "reversed" if oriented[1] < 0 else "unresolved"
        counts[classification] += 1
        events = [int(d * sign < 0) for sign in signs]
        minimum += min(events)
        maximum += max(events)
        details.append(dict(comparison_id=identity, direction=direction, status=classification,
                            oriented_interval=oriented, possible_signs=list(signs),
                            reversal_indicator_bounds=[min(events), max(events)]))
    M = counts["published"]
    _require(counts["candidates"] == M + counts["held"] and M == counts["supported"] + counts["reversed"] + counts["unresolved"], "Panel count identity failed")
    return dict(counts=counts, coverage=M / len(ids),
                reversal_fraction=counts["reversed"] / M if M else None,
                unresolved_fraction=counts["unresolved"] / M if M else None,
                support_fraction=counts["supported"] / M if M else None,
                true_reversal_count_bounds=[minimum, maximum],
                true_reversal_fraction_bounds=[minimum / M, maximum / M] if M else None,
                comparisons=details, scope="Finite frozen comparison panel, conditional on its common simultaneous loss intervals; no candidate independence assumed.")


def paired_rule_difference_bounds(comparison_ids, directions_j, directions_k, records):
    """Bound risk(j)-risk(k) with a common possible sign for each candidate.

    Matched coverage is required. Shared sign uncertainty must not be optimized
    separately for two rules; identical decisions therefore have exact [0,0]
    risk difference even when every interval is unresolved.
    """
    ids = _panel_inputs(comparison_ids, directions_j, records)
    _panel_inputs(ids, directions_k, records)
    j = finite_panel_risk_bounds(ids, directions_j, records)
    k = finite_panel_risk_bounds(ids, directions_k, records)
    M = j["counts"]["published"]
    _require(M == k["counts"]["published"], "Paired rule comparison requires the same achieved publication count")
    lower = upper = 0
    details = []
    for identity in ids:
        dj, dk = ({"a": 1, "b": -1, "hold": 0}[directions[identity]] for directions in [directions_j, directions_k])
        values = [int(dj * z < 0) - int(dk * z < 0) for z in possible_signs(records[identity])]
        lo, hi = min(values), max(values)
        lower += lo
        upper += hi
        details.append(dict(comparison_id=identity, difference_indicator_bounds=[lo, hi]))
    observed = M > 0 and j["counts"]["reversed"] < k["counts"]["reversed"] and j["counts"]["unresolved"] <= k["counts"]["unresolved"]
    bounds = [lower / M, upper / M] if M else None
    supported = observed and upper < 0
    return dict(published_per_rule=M, coverage=M / len(ids), rule_j=j, rule_k=k,
                true_reversal_count_difference_bounds=[lower, upper],
                true_reversal_fraction_difference_bounds=bounds,
                observed_fewer_reversals_without_more_unresolved=observed,
                interval_supported_finite_panel_improvement=supported,
                status="no_publications" if M == 0 else "supported_finite_panel_improvement" if supported else
                       "observed_advantage_not_interval_supported" if observed else "no_declared_improvement",
                comparisons=details,
                scope="Inherits simultaneous paired-loss interval assumptions. This is not a confidence interval for the sampling-dependent unresolved fraction or for a population of unseen tasks.")

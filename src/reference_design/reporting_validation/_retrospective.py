"""Mathematical kernels from the executed retrospective score-bank comparison.

No expression reader, private path, predictor, or retrospective admission bypass
is included. Public entry points add input and assumption contracts.
"""
import hashlib
import itertools
import math
import numpy as np
from scipy.stats import t as student_t
from .reporting_candidate import RULES, coverage_plans
FAMILIES = ("NC", "CM", "TW", "PCA", "RBF", "scGen", "CellOT")
CONDITIONS = ("IFN-beta", "IFN-gamma", "TNF-alpha")
PAIR_INDEX = list(itertools.combinations(range(7), 2))
TAU = 1e-12

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sign(value):
    return int(value > TAU) - int(value < -TAU)

def direction(value):
    return {1: "a", -1: "b", 0: "hold"}[sign(value)]

def roster(contexts):
    return [dict(pair_id=f"{context}|{condition}|{FAMILIES[a]}:{FAMILIES[b]}",
                 context=context, condition=condition, model_a=FAMILIES[a], model_b=FAMILIES[b],
                 model_a_index=a, model_b_index=b)
            for context in contexts for condition in CONDITIONS for a, b in PAIR_INDEX]

def context_matrix(rows, loss, context):
    donors = sorted({r["native_donor"] for r in rows if r["context"] == context})
    index = {(r["native_donor"], r["context"], r["condition"]): i for i, r in enumerate(rows)}
    require(donors, "Empty context")
    require(all((d, context, c) in index for d in donors for c in CONDITIONS),
            "Frozen complete three-condition block missing")
    values = np.stack([np.stack([loss[index[d, context, c]] for c in CONDITIONS]) for d in donors])
    return donors, np.stack([values[..., b] - values[..., a] for a, b in PAIR_INDEX], axis=-1).reshape(len(donors), 3, 64, 21).transpose(0, 2, 1, 3).reshape(len(donors), 64, 63)

def calibration_penalties(loss):
    return np.array([np.max(np.abs((loss[:, :, b] - loss[:, :, a]) -
                                  (loss[:, 0, b] - loss[:, 0, a])[:, None]))
                     for a, b in PAIR_INDEX])

def candidates_for_context(rows, losses, context, penalty, bootstrap_count=9999, seed=20260922):
    donors, matrix = context_matrix(rows, losses, context)
    require(len(donors) >= 2, "Publication context has fewer than two donors")
    comparisons = roster([context])
    anchor = matrix[:, 0].mean(axis=0)
    unit_mean = matrix.mean(axis=1)
    pooled_allocations = matrix.mean(axis=0)
    repeat_mean = unit_mean.mean(axis=0)
    first_two = matrix[:, :2].mean(axis=(0, 1))
    schedule = np.random.default_rng(seed).integers(0, len(donors), size=(bootstrap_count, len(donors)))
    boot = np.empty((bootstrap_count, 63))
    for start in range(0, bootstrap_count, 128):
        stop = min(start + 128, bootstrap_count)
        boot[start:stop] = unit_mean[schedule[start:stop]].mean(axis=1)
    candidates, companions = [], []
    for j, item in enumerate(comparisons):
        for rule in RULES:
            estimate = anchor[j] if rule in ("score_margin", "existing_refara_reporting") else repeat_mean[j]
            d = sign(estimate)
            if rule == "repeated_splitting":
                score = np.sort(d * pooled_allocations[:, j])[math.floor(.05 * 63)]
            elif rule == "independent_unit_resampling":
                score = np.sort(d * boot[:, j])[math.floor(.05 * (bootstrap_count - 1))]
            elif rule == "existing_refara_reporting":
                score = abs(estimate) - penalty[j % 21]
            else:
                score = abs(estimate)
            eligible = bool(d and score > TAU)
            candidates.append(dict(rule_id=rule, pair_id=item["pair_id"], estimate=float(estimate),
                                   candidate_direction=direction(estimate), reliability_score=float(score),
                                   eligible=eligible, reason="eligible" if eligible else "direction_tie" if not d else "score_below_floor"))
        value = float(first_two[j])
        companions.append(dict(rule_id="cross_fitting_first_partition", pair_id=item["pair_id"],
                               estimate=value, candidate_direction=direction(value), reliability_score=abs(value),
                               eligible=bool(sign(value) and abs(value) > TAU),
                               reason="eligible" if abs(value) > TAU else "direction_tie"))
    require(np.allclose(pooled_allocations.mean(axis=0), repeat_mean, rtol=0, atol=1e-12),
            "Repeated mean and full crossfit identity failed")
    return candidates, companions, dict(context=context, donors=donors, repetitions=bootstrap_count,
                                        seed=seed, schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
                                        bootstrap_arithmetic_elements=bootstrap_count * len(donors) * 63,
                                        repeated_mean_equals_full_crossfit=True)

def coverage_protocol():
    return dict(tie_tolerance=TAU, score_floor=0., primary_coverage_target=.5,
                minimum_primary_count=10, minimum_primary_coverage=.25,
                coverage_grid=[.1, .25, .5, .75, .9, 1.])

def companion_prefixes(candidates):
    rows = sorted((r for r in candidates if r["eligible"]), key=lambda r: (-r["reliability_score"], r["pair_id"]))
    result = {0: set()}
    i = 0
    while i < len(rows):
        end = i + 1
        while end < len(rows) and rows[end - 1]["reliability_score"] - rows[end]["reliability_score"] <= TAU:
            end += 1
        result[end] = {r["pair_id"] for r in rows[:end]}
        i = end
    return result

def mean_se(x, axis=0):
    origin = np.take(x, 0, axis=axis)
    mean = origin + (x - np.expand_dims(origin, axis)).mean(axis=axis)
    centered = x - np.expand_dims(mean, axis)
    scale = np.max(np.abs(centered), axis=axis)
    norm = np.divide(centered, np.expand_dims(scale, axis), out=np.zeros_like(centered),
                     where=np.expand_dims(scale > 0, axis))
    n = x.shape[axis]
    return mean, scale * np.sqrt(np.sum(norm * norm, axis=axis) / (n * (n - 1)))

def working_intervals(matrix, ids, *, multiplicity, joint=False, resamples=19999, seed=20260923):
    """Working donor model, not certified native-experiment confidence coverage."""
    require(matrix.ndim == 2 and matrix.shape[1] == len(ids) and matrix.shape[0] >= 2, "Interval axes differ")
    require(not np.isinf(matrix).any(), "Infinite loss margin")
    n, count = matrix.shape
    means, errors = mean_se(matrix)
    critical_t = float(student_t.isf(.05 / (2 * multiplicity), n - 1))
    active = np.isfinite(matrix).all(axis=0) & np.isfinite(means) & np.isfinite(errors) & (errors > 0)
    with np.errstate(over="ignore", invalid="ignore"):
        boot_active = active & np.isfinite(means - critical_t * errors) & np.isfinite(means + critical_t * errors)
    q, schedule_hash, maxima_hash, infinite = None, None, None, 0
    if joint and boot_active.any():
        schedule = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
        maxima = np.empty(resamples)
        selected = matrix[:, boot_active]
        failed = False
        for start in range(0, resamples, 128):
            stop = min(start + 128, resamples)
            bm, bs = mean_se(selected[schedule[start:stop]], axis=1)
            delta = bm - means[boot_active]
            if not (np.isfinite(bm).all() and np.isfinite(bs).all() and np.isfinite(delta).all()):
                failed = True
                break
            vals = np.divide(delta, bs, out=np.zeros_like(delta), where=bs > 0)
            z = (bs == 0) & (delta != 0)
            vals[z] = np.sign(delta[z]) * np.inf
            maxima[start:stop] = np.abs(vals).max(axis=1)
        schedule_hash = hashlib.sha256(schedule.tobytes()).hexdigest()
        if not failed:
            critical = np.sort(maxima)[math.ceil(.95 * (resamples - 1))]
            q = float(critical) if np.isfinite(critical) else None
            maxima_hash = hashlib.sha256(maxima.tobytes()).hexdigest()
            infinite = int(np.isinf(maxima).sum())
    records = {}
    for j, identity in enumerate(ids):
        record = dict(comparison_id=identity, donor_count=n,
                      mean=float(means[j]) if np.isfinite(means[j]) else None,
                      standard_error=float(errors[j]) if np.isfinite(errors[j]) else None,
                      status="unassessable", interval=None, t_interval=None, bootstrap_interval=None,
                      reason="missing_or_nonfinite_margin" if not np.isfinite(matrix[:, j]).all() else "zero_or_nonfinite_SE_without_identity")
        if active[j]:
            with np.errstate(over="ignore", invalid="ignore"):
                width = critical_t * errors[j]
                t_bounds = [float(means[j] - width), float(means[j] + width)]
            if not np.isfinite(t_bounds).all():
                record["reason"] = "nonfinite_t_interval"
                records[identity] = record
                continue
            record["t_interval"] = t_bounds
            if joint and q is None:
                record["reason"] = "nonfinite_joint_bootstrap_critical"
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    width = max(critical_t, q) * errors[j] if joint else width
                    bounds = [float(means[j] - width), float(means[j] + width)]
                if not np.isfinite(bounds).all():
                    record["reason"] = "nonfinite_envelope_interval"
                    records[identity] = record
                    continue
                if joint:
                    record["bootstrap_interval"] = [float(means[j] - q * errors[j]), float(means[j] + q * errors[j])]
                record.update(status="assessable", reason=None,
                              interval=[float(means[j] - width), float(means[j] + width)])
        records[identity] = record
    return dict(records=records, multiplicity_N=multiplicity, donor_count=n, confidence=.95,
                method="bonferroni_t_joint_maxT_envelope" if joint else "bonferroni_t",
                t_critical=critical_t, bootstrap_critical=q, bootstrap_seed=seed if joint else None,
                bootstrap_resamples=resamples if joint else 0, bootstrap_schedule_sha256=schedule_hash,
                bootstrap_maxima_sha256=maxima_hash, infinite_bootstrap_maxima=infinite,
                scope="Retrospective working-independence donor diagnostic. Shared pools may violate assumptions; nominal coverage not validated.")

def pool_sensitivity(rows, losses, donor_split, contexts):
    pool_columns = dict(zip(CONDITIONS, ("pool_beta", "pool_gamma", "pool_tnf")))
    mapping = {r["donor"]: r for r in donor_split}
    pools = sorted({r[k] for r in donor_split for k in ("pool_control", *pool_columns.values())}, key=int)
    result = []
    for context in contexts:
        donors, matrix = context_matrix(rows, losses, context)
        mean_matrix = matrix.mean(axis=1)
        for j, pair in enumerate(roster([context])):
            full = float(mean_matrix[:, j].mean())
            for pool in pools:
                keep = [i for i, d in enumerate(donors) if mapping[d]["pool_control"] != pool and mapping[d][pool_columns[pair["condition"]]] != pool]
                excluded = [d for i, d in enumerate(donors) if i not in keep]
                value = float(mean_matrix[keep, j].mean()) if keep else None
                result.append(dict(pool_id=pool, comparison_id=pair["pair_id"], excluded_donors=",".join(excluded),
                                   remaining_donors=len(keep), full_mean_margin=full, exclusion_mean_margin=value,
                                   full_point_direction=direction(full), exclusion_point_direction=direction(value) if value is not None else "unavailable",
                                   changed_point_sign=(sign(full) != sign(value)) if value is not None else None,
                                   zero_or_unavailable_reason="empty_after_exclusion" if not keep else "numerical_direction_tie" if sign(value) == 0 else ""))
    return result

def work_accounting(p_rows, c_rows, primary_context):
    output = []
    for panel in ("primary", "secondary"):
        selected = [r for r in p_rows if panel == "secondary" or r["context"] == primary_context]
        tasks = len(selected)
        groups = len({(r["native_donor"], r["context"]) for r in selected})
        for rule in (*RULES, "cross_fitting_first_partition"):
            repeats = 1 if rule in ("score_margin", "existing_refara_reporting") else 2 if rule == "cross_fitting_first_partition" else 64
            calibration = len(c_rows) if rule == "existing_refara_reporting" else 0
            cgroups = len({(r["native_donor"], r["context"]) for r in c_rows}) if calibration else 0
            output.append(dict(panel=panel, rule_id=rule, publication_donor_tasks=tasks,
                               calibration_donor_tasks=calibration,
                               unique_physical_controls=16 * (groups + cgroups),
                               unique_physical_treated=8 * (tasks + calibration),
                               conceptual_uncached_state_cell_forwards=tasks * repeats * 8 * 6 + calibration * 64 * 8 * 6,
                               unique_state_cell_forwards_with_point_cache=tasks * (8 if repeats == 1 else 16) * 6 + calibration * 16 * 6,
                               scored_fit_losses=(tasks * repeats + calibration * 64) * 11,
                               source="Six state fits; five direct-effect fits have no control-cell forward. C64 includes its anchor, counted once. These are attributable conceptual costs, not claimed actual runtime; shared caches overlap across rules."))
    return output

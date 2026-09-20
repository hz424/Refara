from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ADAPTER_ID = "sound_life_matching_robustness_v2"
SCHEMA_VERSION = "2.0.0"

SPEC_SNAPSHOT = "SPEC_SNAPSHOT.json"
DESIGN_JSON = "SOUND_LIFE_MAXIMUM_MATCHING_DESIGN_V2.json"
EXHAUSTIVE_TSV = "SOUND_LIFE_MAXIMUM_MATCHINGS_EXHAUSTIVE_12_V2.tsv"
SALT_TSV = "SOUND_LIFE_SALT_WITNESS_MATCHINGS_V2.tsv"
INCLUSION_TSV = "SOUND_LIFE_MATCHING_MEMBERSHIP_BOUNDS_V2.tsv"
UTILITY_TSV = "SOUND_LIFE_CONFIGURATION_UTILITY_BOUNDS_V2.tsv"
CONTRAST_TSV = "SOUND_LIFE_PAIRWISE_CONTRAST_BOUNDS_V2.tsv"
WIN_TSV = "SOUND_LIFE_PAIRWISE_WIN_BOUNDS_V2.tsv"
LEADER_TSV = "SOUND_LIFE_LEADER_ROBUSTNESS_V2.tsv"
HOLM_TSV = "SOUND_LIFE_HOLM_ROBUSTNESS_V2.tsv"
WITNESS_TSV = "SOUND_LIFE_OPTIMIZATION_WITNESSES_V2.tsv"
RESULT_JSON = "SOUND_LIFE_MATCHING_ROBUSTNESS_RESULT_V2.json"
METHODS_MD = "SOUND_LIFE_MATCHING_ROBUSTNESS_METHODS_V2.md"
RECEIPT_JSON = "SOUND_LIFE_MATCHING_ROBUSTNESS_RECEIPT_V2.json"
MANIFEST = "SOUND_LIFE_MATCHING_ROBUSTNESS_MANIFEST_V2.sha256"

PAYLOAD_FILES = (
    SPEC_SNAPSHOT,
    DESIGN_JSON,
    EXHAUSTIVE_TSV,
    SALT_TSV,
    INCLUSION_TSV,
    UTILITY_TSV,
    CONTRAST_TSV,
    WIN_TSV,
    LEADER_TSV,
    HOLM_TSV,
    WITNESS_TSV,
    RESULT_JSON,
    METHODS_MD,
    RECEIPT_JSON,
)

ACQUISITION_HEADER = (
    "donor_id",
    "flu_year",
    "timepoint",
    "cells",
    "sample.sampleKitGuid",
    "sample.sampleKitGuid__n",
    "specimen.specimenGuid",
    "specimen.specimenGuid__n",
    "pipeline.fileGuid",
    "pipeline.fileGuid__n",
    "batch_id",
    "batch_id__n",
    "pool_id",
    "pool_id__n",
    "chip_id",
    "chip_id__n",
    "well_id",
    "well_id__n",
)
ROLE_HEADER = (
    "subject_id",
    "flu_year",
    "role",
    "retain",
    "structural_triplet_complete",
    "hai_d0_identifier_link",
    "hai_d7_identifier_link",
    "donor_wave_count",
    "within_donor_wave_weight_exact",
    "role_donor_weight_exact",
    "role_wave_weight_exact",
)
UTILITY_HEADER = (
    "model_index",
    "model_id",
    "wave_index",
    "donor_id",
    "flu_year",
    "wave_utility_decimal",
    "wave_utility_hex",
)


class RobustnessError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RobustnessError(message)


def byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_receipt() -> dict[str, str]:
    """Describe the active implementation using repository-relative paths.

    The spec's project root deliberately points at the separately supplied
    restricted inputs.  Implementation provenance belongs to this repository
    instead, so it must never be relativized against that external directory.
    """

    import numpy
    import scipy

    implementation = Path(__file__).resolve()
    repository_root = implementation.parents[2]
    require(
        (repository_root / "pyproject.toml").is_file(),
        "repository root is absent",
    )
    files = {
        "source": implementation,
        "test": implementation.parent
        / "tests/test_sound_life_matching_robustness_v2.py",
        "analysis_spec": implementation.parent
        / "SOUND_LIFE_ROBUST_MATCHING_ANALYSIS_SPEC_V2.md",
    }
    for label, path in files.items():
        require(
            path.is_file() and not path.is_symlink(),
            f"implementation provenance file is absent or unsafe: {label}",
        )
    return {
        "source_path": files["source"].relative_to(repository_root).as_posix(),
        "source_sha256": sha256_file(files["source"]),
        "test_path": files["test"].relative_to(repository_root).as_posix(),
        "test_sha256": sha256_file(files["test"]),
        "analysis_spec_path": files["analysis_spec"]
        .relative_to(repository_root)
        .as_posix(),
        "analysis_spec_sha256": sha256_file(files["analysis_spec"]),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def read_tsv(path: Path, header: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(
            tuple(reader.fieldnames or ()) == tuple(header),
            f"header differs: {path}",
        )
        return list(reader)


def write_tsv(
    path: Path, header: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=header, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            require(set(row) == set(header), f"output fields differ: {path.name}")
            writer.writerow({name: row[name] for name in header})


def parse_int(value: Any, name: str, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{name} is not an integer",
    )
    require(value >= minimum, f"{name} is below its minimum")
    return value


def fraction_fields(prefix: str, value: Fraction) -> dict[str, Any]:
    return {
        f"{prefix}_numerator": value.numerator,
        f"{prefix}_denominator": value.denominator,
        f"{prefix}_decimal": format(float(value), ".17g"),
    }


@dataclass(frozen=True)
class Edge:
    donor_id: str
    flu_year: str
    batch_id: str
    pool_id: str
    d0_cells: int
    d7_cells: int

    def key(self) -> tuple[bytes, bytes, bytes, bytes]:
        return (
            byte_key(self.donor_id),
            byte_key(self.batch_id),
            byte_key(self.flu_year),
            byte_key(self.pool_id),
        )

    def text_key(self) -> str:
        return "\x1f".join(
            (self.donor_id, self.batch_id, self.flu_year, self.pool_id)
        )


@dataclass(frozen=True)
class UtilityValue:
    binary64: float
    exact: Fraction


def load_spec(spec_path: Path) -> tuple[dict[str, Any], bytes, Path]:
    spec_path = spec_path.resolve()
    raw = spec_path.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    require(spec.get("schema_version") == SCHEMA_VERSION, "spec version differs")
    require(spec.get("adapter") == ADAPTER_ID, "adapter differs")
    require(
        spec.get("evidence_label")
        == "POST_HOC_OUTCOME_EXPOSED_ROBUST_MATCHING_SENSITIVITY",
        "evidence label differs",
    )
    root_relative = spec.get("project_root_relative_to_spec")
    require(
        isinstance(root_relative, str) and root_relative,
        "project root locator differs",
    )
    project_root = (spec_path.parent / root_relative).resolve()
    require(project_root.is_dir(), "project root is absent")
    graphs = spec.get("graphs")
    require(isinstance(graphs, list) and len(graphs) == 2, "graph roster differs")
    inference = spec.get("inference")
    require(isinstance(inference, dict), "inference object absent")
    require(
        len(inference.get("model_ids_in_order", [])) == 8
        and inference.get("ordered_comparisons") == 56,
        "model/comparison geometry differs",
    )
    boundary = spec.get("authority_boundary")
    require(
        isinstance(boundary, dict)
        and all(value is False for value in boundary.values()),
        "scope record differs",
    )
    return spec, raw, project_root


def input_bindings(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    inputs = spec.get("inputs")
    require(isinstance(inputs, list) and len(inputs) == 3, "input roster differs")
    result: dict[str, dict[str, str]] = {}
    for item in inputs:
        require(isinstance(item, dict), "input binding is not an object")
        role = item.get("role")
        path = item.get("path")
        digest = item.get("sha256")
        require(
            isinstance(role, str) and role not in result,
            "input role differs",
        )
        require(
            isinstance(path, str) and path and not Path(path).is_absolute(),
            "input path is unsafe",
        )
        require(
            isinstance(digest, str) and len(digest) == 64,
            "input hash differs",
        )
        result[role] = {"path": path, "sha256": digest}
    require(
        set(result)
        == {
            "acquisition_metadata",
            "frozen_role_wave_mask",
            "frozen_eval_wave_utility",
        },
        "input roles differ",
    )
    return result


def verify_binding(project_root: Path, binding: Mapping[str, str]) -> dict[str, Any]:
    path = (project_root / binding["path"]).resolve()
    require(path.is_file() and not path.is_symlink(), "bound input is absent or unsafe")
    digest = sha256_file(path)
    require(digest == binding["sha256"], f"bound input hash differs: {binding['path']}")
    return {
        "path": binding["path"],
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def load_metadata(
    project_root: Path, bindings: Mapping[str, Mapping[str, str]]
) -> tuple[list[Edge], set[str], dict[str, Any]]:
    role_rows = read_tsv(
        project_root / bindings["frozen_role_wave_mask"]["path"], ROLE_HEADER
    )
    require(len(role_rows) == 171, "role-wave geometry differs")
    role_by_wave: dict[tuple[str, str], str] = {}
    donors: dict[str, set[str]] = defaultdict(set)
    wave_counts: dict[str, int] = defaultdict(int)
    for row in role_rows:
        key = (row["subject_id"], row["flu_year"])
        require(key not in role_by_wave, "duplicate role wave")
        require(row["role"] in {"TRAIN", "EVAL"}, "role differs")
        require(
            row["retain"] == "TRUE"
            and row["structural_triplet_complete"] == "TRUE",
            "frozen wave admission differs",
        )
        role_by_wave[key] = row["role"]
        donors[row["role"]].add(row["subject_id"])
        wave_counts[row["role"]] += 1
    require(
        {key: len(value) for key, value in donors.items()}
        == {"TRAIN": 30, "EVAL": 59},
        "donor-role geometry differs",
    )
    require(
        dict(wave_counts) == {"EVAL": 112, "TRAIN": 59},
        "wave-role geometry differs",
    )

    acquisition_rows = read_tsv(
        project_root / bindings["acquisition_metadata"]["path"],
        ACQUISITION_HEADER,
    )
    require(len(acquisition_rows) == 523, "acquisition geometry differs")
    by_wave: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in acquisition_rows:
        key = (row["donor_id"], row["flu_year"])
        require(
            row["timepoint"] not in by_wave[key],
            "duplicate acquisition timepoint",
        )
        by_wave[key][row["timepoint"]] = row

    train_batches: set[str] = set()
    candidates: list[Edge] = []
    for donor_id, flu_year in sorted(
        role_by_wave, key=lambda item: (byte_key(item[0]), byte_key(item[1]))
    ):
        rows = by_wave.get((donor_id, flu_year), {})
        require(set(rows) == {"D0", "D7", "D90"}, "acquisition triplet differs")
        d0, d7 = rows["D0"], rows["D7"]
        for row in (d0, d7):
            require(
                row["batch_id__n"] == "1" and row["pool_id__n"] == "1",
                "non-singleton D0/D7 label",
            )
            require(row["batch_id"] and row["pool_id"], "D0/D7 label absent")
        if role_by_wave[(donor_id, flu_year)] == "TRAIN":
            train_batches.update((d0["batch_id"], d7["batch_id"]))
        if (
            role_by_wave[(donor_id, flu_year)] == "EVAL"
            and d0["batch_id"] == d7["batch_id"]
            and d0["pool_id"] == d7["pool_id"]
        ):
            candidates.append(
                Edge(
                    donor_id=donor_id,
                    flu_year=flu_year,
                    batch_id=d0["batch_id"],
                    pool_id=d0["pool_id"],
                    d0_cells=int(d0["cells"]),
                    d7_cells=int(d7["cells"]),
                )
            )
    candidates.sort(key=Edge.key)
    require(len(candidates) == 93, "candidate count differs")
    require(len(train_batches) == 37, "TRAIN batch count differs")
    require(
        len({(edge.donor_id, edge.batch_id) for edge in candidates}) == 93,
        "candidate donor-batch edge is not unique",
    )
    return candidates, train_batches, {
        "frozen_waves": 171,
        "frozen_eval_donors": 59,
        "frozen_eval_waves": 112,
        "train_d0_d7_batches": 37,
    }


def maximum_matching_size(edges: Sequence[Edge]) -> int:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.donor_id].append(edge.batch_id)
    for donor in adjacency:
        adjacency[donor] = sorted(set(adjacency[donor]), key=byte_key)
    batch_to_donor: dict[str, str] = {}

    def augment(donor: str, seen: set[str]) -> bool:
        for batch in adjacency[donor]:
            if batch in seen:
                continue
            seen.add(batch)
            if batch not in batch_to_donor or augment(
                batch_to_donor[batch], seen
            ):
                batch_to_donor[batch] = donor
                return True
        return False

    for donor in sorted(adjacency, key=byte_key):
        augment(donor, set())
    return len(batch_to_donor)


def membership_digest(edges: Sequence[Edge]) -> str:
    payload = "\n".join(
        edge.text_key() for edge in sorted(edges, key=Edge.key)
    ) + "\n"
    return sha256_bytes(payload.encode("utf-8"))


def validate_matching(
    matching: Sequence[Edge], available: Sequence[Edge], target: int
) -> tuple[Edge, ...]:
    result = tuple(sorted(matching, key=Edge.key))
    require(len(result) == target, "matching cardinality differs")
    require(len({edge.donor_id for edge in result}) == target, "donor collision")
    require(len({edge.batch_id for edge in result}) == target, "batch collision")
    require(set(result).issubset(set(available)), "matching contains foreign edge")
    return result


def hash_ordered_maximum_matching(
    edges: Sequence[Edge], salt: str, target: int
) -> tuple[Edge, ...]:
    ordered = sorted(
        edges,
        key=lambda edge: (
            hashlib.sha256(
                (salt + "\0" + edge.text_key()).encode("utf-8")
            ).digest(),
            edge.key(),
        ),
    )
    chosen: list[Edge] = []
    used_donors: set[str] = set()
    used_batches: set[str] = set()
    start = 0
    while len(chosen) < target:
        needed = target - len(chosen)
        accepted = False
        for index in range(start, len(ordered)):
            edge = ordered[index]
            if edge.donor_id in used_donors or edge.batch_id in used_batches:
                continue
            blocked_donors = used_donors | {edge.donor_id}
            blocked_batches = used_batches | {edge.batch_id}
            remaining = [
                candidate
                for candidate in ordered[index + 1 :]
                if candidate.donor_id not in blocked_donors
                and candidate.batch_id not in blocked_batches
            ]
            if 1 + maximum_matching_size(remaining) >= needed:
                chosen.append(edge)
                used_donors.add(edge.donor_id)
                used_batches.add(edge.batch_id)
                start = index + 1
                accepted = True
                break
        require(accepted, "salted maximum matching construction failed")
    return validate_matching(chosen, edges, target)


def enumerate_maximum_matchings(
    edges: Sequence[Edge], target: int, expected_count: int
) -> list[tuple[Edge, ...]]:
    by_batch: dict[str, list[Edge]] = defaultdict(list)
    for edge in edges:
        by_batch[edge.batch_id].append(edge)
    for batch in by_batch:
        by_batch[batch].sort(key=Edge.key)
    batches = tuple(
        sorted(by_batch, key=lambda value: (len(by_batch[value]), byte_key(value)))
    )
    output: dict[str, tuple[Edge, ...]] = {}

    def visit(
        batch_index: int, used_donors: frozenset[str], chosen: tuple[Edge, ...]
    ) -> None:
        if len(chosen) == target:
            matching = validate_matching(chosen, edges, target)
            output[membership_digest(matching)] = matching
            return
        if batch_index == len(batches):
            return
        if len(chosen) + len(batches) - batch_index < target:
            return
        remaining_batches = set(batches[batch_index:])
        remaining_edges = [
            edge
            for edge in edges
            if edge.batch_id in remaining_batches
            and edge.donor_id not in used_donors
        ]
        if len(chosen) + maximum_matching_size(remaining_edges) < target:
            return
        batch = batches[batch_index]
        if len(chosen) + len(batches) - batch_index - 1 >= target:
            visit(batch_index + 1, used_donors, chosen)
        for edge in by_batch[batch]:
            if edge.donor_id not in used_donors:
                visit(
                    batch_index + 1,
                    used_donors | {edge.donor_id},
                    chosen + (edge,),
                )

    visit(0, frozenset(), ())
    require(len(output) == expected_count, "exhaustive matching count differs")
    return [output[digest] for digest in sorted(output)]


@dataclass
class FlowArc:
    destination: int
    reverse_index: int
    capacity: int
    cost: Fraction
    edge_index: int | None


def exact_min_cost_matching(
    edges: Sequence[Edge], target: int, weights: Sequence[Fraction]
) -> tuple[Fraction, tuple[Edge, ...]]:
    require(len(edges) == len(weights), "objective geometry differs")
    donors = tuple(sorted({edge.donor_id for edge in edges}, key=byte_key))
    batches = tuple(sorted({edge.batch_id for edge in edges}, key=byte_key))
    source = 0
    donor_node = {value: index + 1 for index, value in enumerate(donors)}
    batch_offset = 1 + len(donors)
    batch_node = {
        value: batch_offset + index for index, value in enumerate(batches)
    }
    sink = batch_offset + len(batches)
    adjacency: list[list[FlowArc]] = [[] for _ in range(sink + 1)]

    def add_arc(
        origin: int,
        destination: int,
        capacity: int,
        cost: Fraction,
        edge_index: int | None = None,
    ) -> int:
        forward_index = len(adjacency[origin])
        reverse_index = len(adjacency[destination])
        adjacency[origin].append(
            FlowArc(destination, reverse_index, capacity, cost, edge_index)
        )
        adjacency[destination].append(
            FlowArc(origin, forward_index, 0, -cost, None)
        )
        return forward_index

    for donor in donors:
        add_arc(source, donor_node[donor], 1, Fraction(0))
    candidate_arcs: list[tuple[int, int]] = []
    for index, edge in enumerate(edges):
        arc_index = add_arc(
            donor_node[edge.donor_id],
            batch_node[edge.batch_id],
            1,
            weights[index],
            index,
        )
        candidate_arcs.append((donor_node[edge.donor_id], arc_index))
    for batch in batches:
        add_arc(batch_node[batch], sink, 1, Fraction(0))

    total_cost = Fraction(0)
    for _ in range(target):
        distance: list[Fraction | None] = [None] * len(adjacency)
        predecessor: list[tuple[int, int] | None] = [None] * len(adjacency)
        queued = [False] * len(adjacency)
        queue: deque[int] = deque([source])
        distance[source] = Fraction(0)
        queued[source] = True
        while queue:
            origin = queue.popleft()
            queued[origin] = False
            base = distance[origin]
            require(base is not None, "shortest-path state differs")
            for arc_index, arc in enumerate(adjacency[origin]):
                if arc.capacity == 0:
                    continue
                candidate = base + arc.cost
                prior = distance[arc.destination]
                if prior is None or candidate < prior:
                    distance[arc.destination] = candidate
                    predecessor[arc.destination] = (origin, arc_index)
                    if not queued[arc.destination]:
                        queue.append(arc.destination)
                        queued[arc.destination] = True
        require(distance[sink] is not None, "target matching is infeasible")
        total_cost += distance[sink]  # type: ignore[arg-type]
        cursor = sink
        while cursor != source:
            prior = predecessor[cursor]
            require(prior is not None, "augmenting path is incomplete")
            origin, arc_index = prior
            arc = adjacency[origin][arc_index]
            arc.capacity -= 1
            adjacency[cursor][arc.reverse_index].capacity += 1
            cursor = origin

    selected = [
        edges[index]
        for index, (origin, arc_index) in enumerate(candidate_arcs)
        if adjacency[origin][arc_index].capacity == 0
    ]
    matching = validate_matching(selected, edges, target)
    exact = sum(
        (weights[edges.index(edge)] for edge in matching), Fraction(0)
    )
    require(exact == total_cost, "exact min-cost objective replay differs")
    return exact, matching


def optimize_additive(
    edges: Sequence[Edge],
    target: int,
    weights: Sequence[Fraction],
    sense: str,
) -> tuple[Fraction, tuple[Edge, ...]]:
    require(sense in {"MIN", "MAX"}, "optimization sense differs")
    objective = list(weights)
    if sense == "MAX":
        objective = [-value for value in objective]
    _, matching = exact_min_cost_matching(edges, target, objective)
    value = sum(
        (weights[edges.index(edge)] for edge in matching), Fraction(0)
    )
    return value, matching


def load_utilities(
    project_root: Path,
    binding: Mapping[str, str],
    model_ids: Sequence[str],
) -> dict[tuple[str, str, str], UtilityValue]:
    rows = read_tsv(project_root / binding["path"], UTILITY_HEADER)
    require(len(rows) == 848, "utility geometry differs")
    values: dict[tuple[str, str, str], UtilityValue] = {}
    for row in rows:
        model_index = int(row["model_index"])
        require(
            0 <= model_index < len(model_ids)
            and row["model_id"] == model_ids[model_index],
            "utility model roster differs",
        )
        decimal = float(row["wave_utility_decimal"])
        hexadecimal = float.fromhex(row["wave_utility_hex"])
        require(
            decimal.hex() == hexadecimal.hex(),
            "utility decimal/hex serialization differs",
        )
        key = (row["donor_id"], row["flu_year"], row["model_id"])
        require(key not in values, "duplicate utility")
        numerator, denominator = hexadecimal.as_integer_ratio()
        values[key] = UtilityValue(
            binary64=hexadecimal, exact=Fraction(numerator, denominator)
        )
    return values


def complete_family(
    model_ids: Sequence[str],
) -> list[tuple[int, str, int, str]]:
    return [
        (source_index, source, target_index, target)
        for source_index, source in enumerate(model_ids)
        for target_index, target in enumerate(model_ids)
        if source_index != target_index
    ]


def sign_tail(wins: int, non_tied: int) -> Fraction:
    require(0 <= wins <= non_tied, "sign-count geometry differs")
    return Fraction(
        sum(comb(non_tied, index) for index in range(wins, non_tied + 1)),
        2**non_tied,
    )


def holm(
    raw: Sequence[Fraction], alpha: Fraction
) -> tuple[list[Fraction], list[int], list[bool]]:
    order = sorted(range(len(raw)), key=lambda index: (raw[index], index))
    adjusted = [Fraction(0)] * len(raw)
    ranks = [0] * len(raw)
    running = Fraction(0)
    for rank, index in enumerate(order, 1):
        running = max(
            running, min(Fraction(1), raw[index] * (len(raw) - rank + 1))
        )
        adjusted[index] = running
        ranks[index] = rank
    return adjusted, ranks, [value <= alpha for value in adjusted]


def analyze_matching(
    matching: Sequence[Edge],
    model_ids: Sequence[str],
    utilities: Mapping[tuple[str, str, str], UtilityValue],
    alpha: Fraction,
) -> dict[str, Any]:
    sums: list[Fraction] = []
    for model_id in model_ids:
        sums.append(
            sum(
                (
                    utilities[(edge.donor_id, edge.flu_year, model_id)].exact
                    for edge in matching
                ),
                Fraction(0),
            )
        )
    maximum = max(sums)
    leaders = {
        model_ids[index] for index, value in enumerate(sums) if value == maximum
    }
    family = complete_family(model_ids)
    counts: list[tuple[int, int, int]] = []
    raw: list[Fraction] = []
    for _, source, _, target in family:
        differences = [
            utilities[(edge.donor_id, edge.flu_year, source)].binary64
            - utilities[(edge.donor_id, edge.flu_year, target)].binary64
            for edge in matching
        ]
        wins = sum(value > 0.0 for value in differences)
        losses = sum(value < 0.0 for value in differences)
        ties = len(differences) - wins - losses
        counts.append((wins, losses, ties))
        raw.append(sign_tail(wins, wins + losses))
    adjusted, ranks, rejected = holm(raw, alpha)
    return {
        "sums": sums,
        "leaders": leaders,
        "counts": counts,
        "raw": raw,
        "adjusted": adjusted,
        "ranks": ranks,
        "rejected": rejected,
    }


def build_metadata_design(
    spec: dict[str, Any],
    candidates: Sequence[Edge],
    train_batches: set[str],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Edge, ...]],
    dict[str, list[tuple[Edge, ...]]],
    dict[str, list[tuple[Edge, ...]]],
    list[str],
]:
    witness = spec["witness_ensemble"]
    salt_start = parse_int(witness.get("salt_start"), "salt_start")
    salt_count = parse_int(witness.get("salt_count"), "salt_count", 1)
    salt_width = parse_int(witness.get("salt_width"), "salt_width", 1)
    prefix = witness.get("salt_prefix")
    require(isinstance(prefix, str) and prefix, "salt prefix differs")
    salts = [
        f"{prefix}{index:0{salt_width}d}"
        for index in range(salt_start, salt_start + salt_count)
    ]
    require(len(set(salts)) == salt_count, "salt roster is not unique")

    graph_edges: dict[str, tuple[Edge, ...]] = {}
    exhaustive: dict[str, list[tuple[Edge, ...]]] = {}
    salt_matchings: dict[str, list[tuple[Edge, ...]]] = {}
    graph_rows: list[dict[str, Any]] = []
    for graph_spec in spec["graphs"]:
        graph_id = graph_spec["graph_id"]
        remove_train = graph_spec["remove_train_d0_d7_batches"]
        require(
            isinstance(graph_id, str) and isinstance(remove_train, bool),
            "graph spec differs",
        )
        edges = tuple(
            edge
            for edge in candidates
            if not remove_train or edge.batch_id not in train_batches
        )
        donors = {edge.donor_id for edge in edges}
        batches = {edge.batch_id for edge in edges}
        target = maximum_matching_size(edges)
        require(
            len(edges) == graph_spec["expected_candidate_waves"],
            f"candidate-wave count differs: {graph_id}",
        )
        require(
            len(donors) == graph_spec["expected_candidate_donors"],
            f"candidate-donor count differs: {graph_id}",
        )
        require(
            len(batches) == graph_spec["expected_candidate_batches"],
            f"candidate-batch count differs: {graph_id}",
        )
        require(
            target == graph_spec["expected_maximum_cardinality"],
            f"maximum cardinality differs: {graph_id}",
        )
        graph_edges[graph_id] = edges
        salt_matchings[graph_id] = [
            hash_ordered_maximum_matching(edges, salt, target) for salt in salts
        ]
        exhaustive_count: int | None = None
        if graph_spec["analysis_mode"] == "EXHAUSTIVE_ENUMERATION":
            expected = graph_spec.get("expected_maximum_matching_count")
            require(
                isinstance(expected, int) and expected > 0,
                "expected exhaustive count differs",
            )
            exhaustive[graph_id] = enumerate_maximum_matchings(
                edges, target, expected
            )
            exhaustive_count = len(exhaustive[graph_id])
        else:
            require(
                graph_spec["analysis_mode"] == "EXACT_ADDITIVE_OPTIMIZATION",
                "analysis mode differs",
            )
        candidate_payload = "\n".join(edge.text_key() for edge in edges) + "\n"
        salt_digest_payload = "\n".join(
            membership_digest(matching)
            for matching in salt_matchings[graph_id]
        ) + "\n"
        exhaustive_digest = None
        if graph_id in exhaustive:
            exhaustive_digest = sha256_bytes(
                (
                    "\n".join(
                        membership_digest(matching)
                        for matching in exhaustive[graph_id]
                    )
                    + "\n"
                ).encode("ascii")
            )
        graph_rows.append(
            {
                "graph_id": graph_id,
                "remove_train_d0_d7_batches": remove_train,
                "candidate_waves": len(edges),
                "candidate_donors": len(donors),
                "candidate_batches": len(batches),
                "maximum_cardinality": target,
                "analysis_mode": graph_spec["analysis_mode"],
                "candidate_edge_set_sha256": sha256_bytes(
                    candidate_payload.encode("utf-8")
                ),
                "salt_witness_count": len(salts),
                "salt_witness_sequence_sha256": sha256_bytes(
                    salt_digest_payload.encode("ascii")
                ),
                "exhaustive_maximum_matching_count": exhaustive_count,
                "exhaustive_membership_set_sha256": exhaustive_digest,
            }
        )
    design = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "SOUND_LIFE_MAXIMUM_MATCHING_DESIGN_V2",
        "adapter": ADAPTER_ID,
        "evidence_label": "OUTCOME_FREE_METADATA_GRAPH_DESIGN",
        "metadata_frozen_before_utility_hash_or_open": True,
        "utility_used_to_construct_candidate_graphs": False,
        "graphs": graph_rows,
        "salt_witnesses": {
            "algorithm": witness["algorithm"],
            "salt_count": salt_count,
            "first_salt": salts[0],
            "last_salt": salts[-1],
            "salt_set_sha256": sha256_bytes(
                ("\n".join(salts) + "\n").encode("utf-8")
            ),
            "frequency_interpretation": witness["frequency_interpretation"],
        },
    }
    return design, graph_edges, exhaustive, salt_matchings, salts


def leader_maximin_milp(
    edges: Sequence[Edge],
    target: int,
    model_id: str,
    model_ids: Sequence[str],
    utilities: Mapping[tuple[str, str, str], UtilityValue],
    time_limit: int,
    tolerance: float,
) -> dict[str, Any]:
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix

    require(scipy.__version__ == "1.15.3", "SciPy version differs")
    edge_count = len(edges)
    donors = tuple(sorted({edge.donor_id for edge in edges}, key=byte_key))
    batches = tuple(sorted({edge.batch_id for edge in edges}, key=byte_key))
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for donor in donors:
        rows.append(
            [float(edge.donor_id == donor) for edge in edges] + [0.0]
        )
        lower.append(-np.inf)
        upper.append(1.0)
    for batch in batches:
        rows.append(
            [float(edge.batch_id == batch) for edge in edges] + [0.0]
        )
        lower.append(-np.inf)
        upper.append(1.0)
    rows.append([1.0] * edge_count + [0.0])
    lower.append(float(target))
    upper.append(float(target))
    coefficient_max = 0.0
    for competitor in model_ids:
        if competitor == model_id:
            continue
        coefficients = [
            float(
                utilities[(edge.donor_id, edge.flu_year, model_id)].exact
                - utilities[(edge.donor_id, edge.flu_year, competitor)].exact
            )
            for edge in edges
        ]
        coefficient_max = max(coefficient_max, *(abs(value) for value in coefficients))
        rows.append(coefficients + [-1.0])
        lower.append(0.0)
        upper.append(np.inf)
    z_limit = max(1.0, 2.0 * target * coefficient_max)
    objective = np.zeros(edge_count + 1, dtype=float)
    objective[-1] = -1.0
    integrality = np.r_[np.ones(edge_count, dtype=int), 0]
    bounds = Bounds(
        np.r_[np.zeros(edge_count), -z_limit],
        np.r_[np.ones(edge_count), z_limit],
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            csr_matrix(np.asarray(rows, dtype=float)),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"mip_rel_gap": 0.0, "time_limit": float(time_limit)},
    )
    if result.x is None:
        return {
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "possible": None,
            "certificate": "UNRESOLVED_NO_INTEGER_WITNESS",
            "matching": None,
            "witness_minimum_exact_margin": None,
            "solver_maximin_decimal": None,
            "solver_maximin_upper_bound_decimal": None,
            "mip_gap_decimal": None,
        }
    chosen = tuple(
        edges[index]
        for index, value in enumerate(result.x[:edge_count])
        if value > 0.5
    )
    matching = validate_matching(chosen, edges, target)
    margins = [
        sum(
            (
                utilities[(edge.donor_id, edge.flu_year, model_id)].exact
                - utilities[(edge.donor_id, edge.flu_year, competitor)].exact
                for edge in matching
            ),
            Fraction(0),
        )
        for competitor in model_ids
        if competitor != model_id
    ]
    minimum_margin = min(margins)
    dual_bound = getattr(result, "mip_dual_bound", None)
    z_upper = None if dual_bound is None else -float(dual_bound)
    if minimum_margin >= 0:
        possible: bool | None = True
        certificate = "EXACT_RATIONAL_FEASIBLE_WITNESS"
    elif (
        result.status == 0
        and z_upper is not None
        and z_upper < -tolerance
    ):
        possible = False
        certificate = "HIGHS_OPTIMAL_MAXIMIN_UPPER_BOUND_BELOW_ZERO"
    else:
        possible = None
        certificate = "UNRESOLVED_NUMERICAL_BOUNDARY"
    return {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "possible": possible,
        "certificate": certificate,
        "matching": matching,
        "witness_minimum_exact_margin": minimum_margin,
        "solver_maximin_decimal": format(float(result.x[-1]), ".17g"),
        "solver_maximin_upper_bound_decimal": (
            "" if z_upper is None else format(z_upper, ".17g")
        ),
        "mip_gap_decimal": (
            ""
            if getattr(result, "mip_gap", None) is None
            else format(float(result.mip_gap), ".17g")
        ),
    }


def add_witness(
    rows: list[dict[str, Any]],
    graph_id: str,
    objective_type: str,
    objective_id: str,
    sense: str,
    matching: Sequence[Edge],
) -> str:
    digest = membership_digest(matching)
    witness_index = len({row["witness_id"] for row in rows})
    witness_id = f"W{witness_index:05d}"
    for selection_rank, edge in enumerate(sorted(matching, key=Edge.key)):
        rows.append(
            {
                "witness_id": witness_id,
                "graph_id": graph_id,
                "objective_type": objective_type,
                "objective_id": objective_id,
                "sense": sense,
                "membership_sha256": digest,
                "selection_rank": selection_rank,
                "donor_id": edge.donor_id,
                "flu_year": edge.flu_year,
                "batch_id": edge.batch_id,
                "pool_id": edge.pool_id,
            }
        )
    return digest


def direct_extrema(
    matchings: Sequence[Sequence[Edge]],
    weights: Mapping[Edge, Fraction],
) -> tuple[Fraction, Fraction]:
    values = [
        sum((weights[edge] for edge in matching), Fraction(0))
        for matching in matchings
    ]
    return min(values), max(values)


def run_analysis(
    spec: dict[str, Any],
    design: dict[str, Any],
    graph_edges: Mapping[str, tuple[Edge, ...]],
    exhaustive: Mapping[str, list[tuple[Edge, ...]]],
    salt_matchings: Mapping[str, list[tuple[Edge, ...]]],
    salts: Sequence[str],
    utilities: Mapping[tuple[str, str, str], UtilityValue],
) -> dict[str, Any]:
    model_ids = tuple(spec["inference"]["model_ids_in_order"])
    family = complete_family(model_ids)
    require(len(family) == 56, "ordered comparison family differs")
    alpha = Fraction(
        spec["inference"]["alpha_numerator"],
        spec["inference"]["alpha_denominator"],
    )
    tolerance = float(
        spec["optimization"]["leader_certification_absolute_margin"]
    )
    time_limit = int(spec["optimization"]["leader_time_limit_seconds"])

    exhaustive_rows: list[dict[str, Any]] = []
    salt_rows: list[dict[str, Any]] = []
    inclusion_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    win_rows: list[dict[str, Any]] = []
    leader_rows: list[dict[str, Any]] = []
    holm_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    graph_results: list[dict[str, Any]] = []

    graph_spec_by_id = {row["graph_id"]: row for row in spec["graphs"]}
    contrast_bounds: dict[tuple[str, str, str], tuple[Fraction, Fraction]] = {}
    win_bounds: dict[
        tuple[str, int], tuple[int, int, int, int]
    ] = {}
    salt_analyses: dict[str, list[dict[str, Any]]] = {}
    exhaustive_analyses: dict[str, list[dict[str, Any]]] = {}

    for graph in design["graphs"]:
        graph_id = graph["graph_id"]
        edges = graph_edges[graph_id]
        target = int(graph["maximum_cardinality"])
        if graph_id in exhaustive:
            exhaustive_analyses[graph_id] = [
                analyze_matching(matching, model_ids, utilities, alpha)
                for matching in exhaustive[graph_id]
            ]
            for matching_index, matching in enumerate(exhaustive[graph_id]):
                digest = membership_digest(matching)
                for selection_rank, edge in enumerate(matching):
                    exhaustive_rows.append(
                        {
                            "graph_id": graph_id,
                            "matching_index": matching_index,
                            "membership_sha256": digest,
                            "selection_rank": selection_rank,
                            "donor_id": edge.donor_id,
                            "flu_year": edge.flu_year,
                            "batch_id": edge.batch_id,
                            "pool_id": edge.pool_id,
                        }
                    )
        salt_analyses[graph_id] = [
            analyze_matching(matching, model_ids, utilities, alpha)
            for matching in salt_matchings[graph_id]
        ]
        for salt_index, (salt, matching) in enumerate(
            zip(salts, salt_matchings[graph_id])
        ):
            digest = membership_digest(matching)
            for selection_rank, edge in enumerate(matching):
                salt_rows.append(
                    {
                        "graph_id": graph_id,
                        "salt_index": salt_index,
                        "salt": salt,
                        "membership_sha256": digest,
                        "selection_rank": selection_rank,
                        "donor_id": edge.donor_id,
                        "flu_year": edge.flu_year,
                        "batch_id": edge.batch_id,
                        "pool_id": edge.pool_id,
                    }
                )

        for candidate_index, edge in enumerate(edges):
            weights = [
                Fraction(int(candidate == edge)) for candidate in edges
            ]
            minimum, _ = optimize_additive(edges, target, weights, "MIN")
            maximum, _ = optimize_additive(edges, target, weights, "MAX")
            require(
                minimum.denominator == maximum.denominator == 1
                and minimum in {0, 1}
                and maximum in {0, 1}
                and minimum <= maximum,
                "membership bounds differ",
            )
            inclusion_rows.append(
                {
                    "graph_id": graph_id,
                    "candidate_index": candidate_index,
                    "donor_id": edge.donor_id,
                    "flu_year": edge.flu_year,
                    "batch_id": edge.batch_id,
                    "pool_id": edge.pool_id,
                    "inclusion_min": int(minimum),
                    "inclusion_max": int(maximum),
                    "classification": (
                        "INVARIANT_INCLUDED"
                        if minimum == maximum == 1
                        else "IMPOSSIBLE_INCLUDED"
                        if minimum == maximum == 0
                        else "VARIABLE_INCLUDED"
                    ),
                    "optimization_scope": "ALL_MAXIMUM_CARDINALITY_MATCHINGS",
                }
            )

        for model_index, model_id in enumerate(model_ids):
            weights = [
                utilities[(edge.donor_id, edge.flu_year, model_id)].exact
                for edge in edges
            ]
            minimum, minimum_matching = optimize_additive(
                edges, target, weights, "MIN"
            )
            maximum, maximum_matching = optimize_additive(
                edges, target, weights, "MAX"
            )
            require(minimum <= maximum, "utility bounds are incoherent")
            if graph_id in exhaustive:
                mapping = dict(zip(edges, weights))
                require(
                    direct_extrema(exhaustive[graph_id], mapping)
                    == (minimum, maximum),
                    "exhaustive utility cross-check differs",
                )
            min_digest = add_witness(
                witness_rows,
                graph_id,
                "CONFIGURATION_UTILITY",
                model_id,
                "MIN",
                minimum_matching,
            )
            max_digest = add_witness(
                witness_rows,
                graph_id,
                "CONFIGURATION_UTILITY",
                model_id,
                "MAX",
                maximum_matching,
            )
            row = {
                "graph_id": graph_id,
                "model_index": model_index,
                "model_id": model_id,
                "matching_cardinality": target,
                **fraction_fields("utility_sum_min", minimum),
                **fraction_fields("utility_sum_max", maximum),
                **fraction_fields("equal_wave_mean_min", minimum / target),
                **fraction_fields("equal_wave_mean_max", maximum / target),
                "minimum_witness_sha256": min_digest,
                "maximum_witness_sha256": max_digest,
                "bound_type": "EXACT_OVER_ALL_MAXIMUM_MATCHINGS",
            }
            utility_rows.append(row)

        for comparison_index, (
            source_index,
            source,
            target_index,
            comparison_target,
        ) in enumerate(family):
            contrast_weights = [
                utilities[(edge.donor_id, edge.flu_year, source)].exact
                - utilities[
                    (edge.donor_id, edge.flu_year, comparison_target)
                ].exact
                for edge in edges
            ]
            contrast_min, contrast_min_matching = optimize_additive(
                edges, target, contrast_weights, "MIN"
            )
            contrast_max, contrast_max_matching = optimize_additive(
                edges, target, contrast_weights, "MAX"
            )
            require(
                contrast_min <= contrast_max,
                "contrast bounds are incoherent",
            )
            if graph_id in exhaustive:
                mapping = dict(zip(edges, contrast_weights))
                require(
                    direct_extrema(exhaustive[graph_id], mapping)
                    == (contrast_min, contrast_max),
                    "exhaustive contrast cross-check differs",
                )
            contrast_bounds[(graph_id, source, comparison_target)] = (
                contrast_min,
                contrast_max,
            )
            contrast_id = f"{source}__VS__{comparison_target}"
            min_digest = add_witness(
                witness_rows,
                graph_id,
                "PAIRWISE_CONTRAST",
                contrast_id,
                "MIN",
                contrast_min_matching,
            )
            max_digest = add_witness(
                witness_rows,
                graph_id,
                "PAIRWISE_CONTRAST",
                contrast_id,
                "MAX",
                contrast_max_matching,
            )
            contrast_rows.append(
                {
                    "graph_id": graph_id,
                    "comparison_index": comparison_index,
                    "source_model_index": source_index,
                    "source_model_id": source,
                    "target_model_index": target_index,
                    "target_model_id": comparison_target,
                    "matching_cardinality": target,
                    **fraction_fields("contrast_sum_min", contrast_min),
                    **fraction_fields("contrast_sum_max", contrast_max),
                    **fraction_fields(
                        "equal_wave_mean_contrast_min",
                        contrast_min / target,
                    ),
                    **fraction_fields(
                        "equal_wave_mean_contrast_max",
                        contrast_max / target,
                    ),
                    "minimum_witness_sha256": min_digest,
                    "maximum_witness_sha256": max_digest,
                    "bound_type": "EXACT_OVER_ALL_MAXIMUM_MATCHINGS",
                }
            )

            win_weights = [
                Fraction(
                    int(
                        utilities[
                            (edge.donor_id, edge.flu_year, source)
                        ].binary64
                        > utilities[
                            (
                                edge.donor_id,
                                edge.flu_year,
                                comparison_target,
                            )
                        ].binary64
                    )
                )
                for edge in edges
            ]
            tie_weights = [
                Fraction(
                    int(
                        utilities[
                            (edge.donor_id, edge.flu_year, source)
                        ].binary64
                        == utilities[
                            (
                                edge.donor_id,
                                edge.flu_year,
                                comparison_target,
                            )
                        ].binary64
                    )
                )
                for edge in edges
            ]
            win_min, win_min_matching = optimize_additive(
                edges, target, win_weights, "MIN"
            )
            win_max, win_max_matching = optimize_additive(
                edges, target, win_weights, "MAX"
            )
            tie_min, _ = optimize_additive(
                edges, target, tie_weights, "MIN"
            )
            tie_max, _ = optimize_additive(
                edges, target, tie_weights, "MAX"
            )
            require(
                all(
                    value.denominator == 1
                    for value in (win_min, win_max, tie_min, tie_max)
                ),
                "integer sign bounds differ",
            )
            require(
                tie_min == tie_max,
                "literal-tie count varies; raw-p bound logic must be extended",
            )
            if graph_id in exhaustive:
                win_mapping = dict(zip(edges, win_weights))
                tie_mapping = dict(zip(edges, tie_weights))
                require(
                    direct_extrema(exhaustive[graph_id], win_mapping)
                    == (win_min, win_max)
                    and direct_extrema(exhaustive[graph_id], tie_mapping)
                    == (tie_min, tie_max),
                    "exhaustive sign-bound cross-check differs",
                )
            win_min_int = int(win_min)
            win_max_int = int(win_max)
            ties = int(tie_min)
            win_bounds[(graph_id, comparison_index)] = (
                win_min_int,
                win_max_int,
                ties,
                ties,
            )
            min_digest = add_witness(
                witness_rows,
                graph_id,
                "PAIRWISE_WINS",
                contrast_id,
                "MIN",
                win_min_matching,
            )
            max_digest = add_witness(
                witness_rows,
                graph_id,
                "PAIRWISE_WINS",
                contrast_id,
                "MAX",
                win_max_matching,
            )
            non_tied = target - ties
            raw_best = sign_tail(win_max_int, non_tied)
            raw_worst = sign_tail(win_min_int, non_tied)
            win_rows.append(
                {
                    "graph_id": graph_id,
                    "comparison_index": comparison_index,
                    "source_model_index": source_index,
                    "source_model_id": source,
                    "target_model_index": target_index,
                    "target_model_id": comparison_target,
                    "matching_cardinality": target,
                    "source_wins_min": win_min_int,
                    "source_wins_max": win_max_int,
                    "source_losses_min": non_tied - win_max_int,
                    "source_losses_max": non_tied - win_min_int,
                    "literal_ties_min": ties,
                    "literal_ties_max": ties,
                    **fraction_fields("raw_p_best", raw_best),
                    **fraction_fields("raw_p_worst", raw_worst),
                    "minimum_win_witness_sha256": min_digest,
                    "maximum_win_witness_sha256": max_digest,
                    "bound_type": "EXACT_OVER_ALL_MAXIMUM_MATCHINGS",
                }
            )

    for graph in design["graphs"]:
        graph_id = graph["graph_id"]
        edges = graph_edges[graph_id]
        target = int(graph["maximum_cardinality"])
        for model_index, model_id in enumerate(model_ids):
            necessary = all(
                contrast_bounds[(graph_id, model_id, competitor)][0] >= 0
                for competitor in model_ids
                if competitor != model_id
            )
            if graph_id in exhaustive:
                analyses = exhaustive_analyses[graph_id]
                leader_count = sum(
                    model_id in analysis["leaders"] for analysis in analyses
                )
                possible: bool | None = leader_count > 0
                necessary = leader_count == len(analyses)
                certificate = "EXHAUSTIVE_144_MAXIMUM_MATCHINGS"
                solver_status = ""
                solver_maximin = ""
                solver_upper = ""
                mip_gap = ""
                witness_margin = None
                possible_matching = next(
                    (
                        matching
                        for matching, analysis in zip(
                            exhaustive[graph_id], analyses
                        )
                        if model_id in analysis["leaders"]
                    ),
                    None,
                )
            else:
                leader_count = ""
                solved = leader_maximin_milp(
                    edges,
                    target,
                    model_id,
                    model_ids,
                    utilities,
                    time_limit,
                    tolerance,
                )
                possible = solved["possible"]
                certificate = solved["certificate"]
                solver_status = solved["solver_status"]
                solver_maximin = solved["solver_maximin_decimal"]
                solver_upper = solved[
                    "solver_maximin_upper_bound_decimal"
                ]
                mip_gap = solved["mip_gap_decimal"]
                witness_margin = solved["witness_minimum_exact_margin"]
                possible_matching = solved["matching"]
            if necessary:
                require(possible is True, "necessary leader is not possible")
                classification = "NECESSARY_LEADER"
            elif possible is True:
                classification = "POSSIBLE_NOT_NECESSARY_LEADER"
            elif possible is False:
                classification = "IMPOSSIBLE_LEADER"
            else:
                classification = "UNRESOLVED_LEADER_POSSIBILITY"
            witness_digest = ""
            if possible_matching is not None:
                witness_digest = add_witness(
                    witness_rows,
                    graph_id,
                    "LEADER_FEASIBILITY",
                    model_id,
                    "POSSIBLE",
                    possible_matching,
                )
            margin_fields = (
                {
                    "witness_minimum_margin_numerator": "",
                    "witness_minimum_margin_denominator": "",
                    "witness_minimum_margin_decimal": "",
                }
                if witness_margin is None
                else fraction_fields("witness_minimum_margin", witness_margin)
            )
            leader_rows.append(
                {
                    "graph_id": graph_id,
                    "model_index": model_index,
                    "model_id": model_id,
                    "possible_leader": (
                        "" if possible is None else str(possible).lower()
                    ),
                    "necessary_leader": str(necessary).lower(),
                    "classification": classification,
                    "certificate": certificate,
                    "exhaustive_leader_count": leader_count,
                    "exhaustive_matching_count": (
                        len(exhaustive[graph_id])
                        if graph_id in exhaustive
                        else ""
                    ),
                    "solver_status": solver_status,
                    "solver_maximin_decimal": solver_maximin,
                    "solver_maximin_upper_bound_decimal": solver_upper,
                    "mip_gap_decimal": mip_gap,
                    **margin_fields,
                    "possible_witness_sha256": witness_digest,
                }
            )

        for comparison_index, (
            source_index,
            source,
            target_index,
            comparison_target,
        ) in enumerate(family):
            win_min, win_max, tie_min, tie_max = win_bounds[
                (graph_id, comparison_index)
            ]
            require(tie_min == tie_max, "tie bounds differ")
            non_tied = target - tie_min
            raw_best = sign_tail(win_max, non_tied)
            raw_worst = sign_tail(win_min, non_tied)
            rejecting_digest = ""
            nonrejecting_digest = ""
            if graph_id in exhaustive:
                analyses = exhaustive_analyses[graph_id]
                decisions = [
                    bool(analysis["rejected"][comparison_index])
                    for analysis in analyses
                ]
                rejected_count = sum(decisions)
                if rejected_count == len(decisions):
                    possible_rejection: bool | None = True
                    necessary_rejection: bool | None = True
                    classification = "INVARIANT_REJECTED_EXHAUSTIVE"
                elif rejected_count == 0:
                    possible_rejection = False
                    necessary_rejection = False
                    classification = "IMPOSSIBLE_REJECTION_EXHAUSTIVE"
                else:
                    possible_rejection = True
                    necessary_rejection = False
                    classification = "POSSIBLE_NOT_NECESSARY_EXHAUSTIVE"
                rejecting = next(
                    (
                        matching
                        for matching, decision in zip(
                            exhaustive[graph_id], decisions
                        )
                        if decision
                    ),
                    None,
                )
                nonrejecting = next(
                    (
                        matching
                        for matching, decision in zip(
                            exhaustive[graph_id], decisions
                        )
                        if not decision
                    ),
                    None,
                )
                if rejecting is not None:
                    rejecting_digest = membership_digest(rejecting)
                if nonrejecting is not None:
                    nonrejecting_digest = membership_digest(nonrejecting)
                witness_reject_count = ""
                witness_count = ""
                certificate = "EXHAUSTIVE_144_MAXIMUM_MATCHINGS"
            else:
                analyses = salt_analyses[graph_id]
                decisions = [
                    bool(analysis["rejected"][comparison_index])
                    for analysis in analyses
                ]
                witness_reject_count = sum(decisions)
                witness_count = len(decisions)
                rejecting = next(
                    (
                        matching
                        for matching, decision in zip(
                            salt_matchings[graph_id], decisions
                        )
                        if decision
                    ),
                    None,
                )
                nonrejecting = next(
                    (
                        matching
                        for matching, decision in zip(
                            salt_matchings[graph_id], decisions
                        )
                        if not decision
                    ),
                    None,
                )
                if rejecting is not None:
                    rejecting_digest = membership_digest(rejecting)
                if nonrejecting is not None:
                    nonrejecting_digest = membership_digest(nonrejecting)
                if raw_worst * len(family) <= alpha:
                    possible_rejection = True
                    necessary_rejection = True
                    classification = (
                        "INVARIANT_REJECTED_BONFERRONI_CERTIFIED"
                    )
                    certificate = (
                        "WORST_RAW_P_TIMES_56_LE_ALPHA_FOR_ALL_MATCHINGS"
                    )
                elif raw_best > alpha:
                    possible_rejection = False
                    necessary_rejection = False
                    classification = (
                        "IMPOSSIBLE_REJECTION_RAW_P_CERTIFIED"
                    )
                    certificate = (
                        "BEST_RAW_P_GT_ALPHA_FOR_ALL_MATCHINGS"
                    )
                elif rejecting is not None and nonrejecting is not None:
                    possible_rejection = True
                    necessary_rejection = False
                    classification = (
                        "POSSIBLE_NOT_NECESSARY_EXPLICIT_WITNESSES"
                    )
                    certificate = (
                        "REJECTING_AND_NONREJECTING_MAXIMUM_MATCHING_WITNESSES"
                    )
                elif rejecting is not None:
                    possible_rejection = True
                    necessary_rejection = None
                    classification = (
                        "POSSIBLE_REJECTION_NECESSITY_UNRESOLVED"
                    )
                    certificate = (
                        "REJECTING_WITNESS_ONLY_OPTIMIZATION_BOUND_INSUFFICIENT"
                    )
                else:
                    possible_rejection = None
                    necessary_rejection = False
                    classification = (
                        "UNRESOLVED_POSSIBILITY_NONREJECTION_WITNESS"
                    )
                    certificate = (
                        "NONREJECTING_WITNESS_ONLY_OPTIMIZATION_BOUND_INSUFFICIENT"
                    )
            holm_rows.append(
                {
                    "graph_id": graph_id,
                    "comparison_index": comparison_index,
                    "source_model_index": source_index,
                    "source_model_id": source,
                    "target_model_index": target_index,
                    "target_model_id": comparison_target,
                    "possible_holm_rejection": (
                        ""
                        if possible_rejection is None
                        else str(possible_rejection).lower()
                    ),
                    "necessary_holm_rejection": (
                        ""
                        if necessary_rejection is None
                        else str(necessary_rejection).lower()
                    ),
                    "classification": classification,
                    "certificate": certificate,
                    **fraction_fields("raw_p_best", raw_best),
                    **fraction_fields("raw_p_worst", raw_worst),
                    "witness_rejection_count": witness_reject_count,
                    "witness_matching_count": witness_count,
                    "rejecting_witness_sha256": rejecting_digest,
                    "nonrejecting_witness_sha256": nonrejecting_digest,
                    "frequency_interpretation": (
                        "NOT_APPLICABLE_EXHAUSTIVE_COMBINATORIAL_SET"
                        if graph_id in exhaustive
                        else "MEMBERSHIP_WITNESSES_ONLY_NOT_NATURAL_PROBABILITY"
                    ),
                }
            )

        leader_class_counts: dict[str, int] = defaultdict(int)
        for row in leader_rows:
            if row["graph_id"] == graph_id:
                leader_class_counts[str(row["classification"])] += 1
        holm_class_counts: dict[str, int] = defaultdict(int)
        for row in holm_rows:
            if row["graph_id"] == graph_id:
                holm_class_counts[str(row["classification"])] += 1
        graph_results.append(
            {
                **graph,
                "leader_classification_counts": dict(
                    sorted(leader_class_counts.items())
                ),
                "holm_classification_counts": dict(
                    sorted(holm_class_counts.items())
                ),
                "exact_additive_bounds_complete": True,
                "all_maximum_matchings_exhaustively_enumerated": (
                    graph_id in exhaustive
                ),
            }
        )

    return {
        "exhaustive_rows": exhaustive_rows,
        "salt_rows": salt_rows,
        "inclusion_rows": inclusion_rows,
        "utility_rows": utility_rows,
        "contrast_rows": contrast_rows,
        "win_rows": win_rows,
        "leader_rows": leader_rows,
        "holm_rows": holm_rows,
        "witness_rows": witness_rows,
        "graph_results": graph_results,
    }


EXHAUSTIVE_HEADER = (
    "graph_id",
    "matching_index",
    "membership_sha256",
    "selection_rank",
    "donor_id",
    "flu_year",
    "batch_id",
    "pool_id",
)
SALT_HEADER = (
    "graph_id",
    "salt_index",
    "salt",
    "membership_sha256",
    "selection_rank",
    "donor_id",
    "flu_year",
    "batch_id",
    "pool_id",
)
INCLUSION_HEADER = (
    "graph_id",
    "candidate_index",
    "donor_id",
    "flu_year",
    "batch_id",
    "pool_id",
    "inclusion_min",
    "inclusion_max",
    "classification",
    "optimization_scope",
)
UTILITY_OUTPUT_HEADER = (
    "graph_id",
    "model_index",
    "model_id",
    "matching_cardinality",
    "utility_sum_min_numerator",
    "utility_sum_min_denominator",
    "utility_sum_min_decimal",
    "utility_sum_max_numerator",
    "utility_sum_max_denominator",
    "utility_sum_max_decimal",
    "equal_wave_mean_min_numerator",
    "equal_wave_mean_min_denominator",
    "equal_wave_mean_min_decimal",
    "equal_wave_mean_max_numerator",
    "equal_wave_mean_max_denominator",
    "equal_wave_mean_max_decimal",
    "minimum_witness_sha256",
    "maximum_witness_sha256",
    "bound_type",
)
CONTRAST_HEADER = (
    "graph_id",
    "comparison_index",
    "source_model_index",
    "source_model_id",
    "target_model_index",
    "target_model_id",
    "matching_cardinality",
    "contrast_sum_min_numerator",
    "contrast_sum_min_denominator",
    "contrast_sum_min_decimal",
    "contrast_sum_max_numerator",
    "contrast_sum_max_denominator",
    "contrast_sum_max_decimal",
    "equal_wave_mean_contrast_min_numerator",
    "equal_wave_mean_contrast_min_denominator",
    "equal_wave_mean_contrast_min_decimal",
    "equal_wave_mean_contrast_max_numerator",
    "equal_wave_mean_contrast_max_denominator",
    "equal_wave_mean_contrast_max_decimal",
    "minimum_witness_sha256",
    "maximum_witness_sha256",
    "bound_type",
)
WIN_HEADER = (
    "graph_id",
    "comparison_index",
    "source_model_index",
    "source_model_id",
    "target_model_index",
    "target_model_id",
    "matching_cardinality",
    "source_wins_min",
    "source_wins_max",
    "source_losses_min",
    "source_losses_max",
    "literal_ties_min",
    "literal_ties_max",
    "raw_p_best_numerator",
    "raw_p_best_denominator",
    "raw_p_best_decimal",
    "raw_p_worst_numerator",
    "raw_p_worst_denominator",
    "raw_p_worst_decimal",
    "minimum_win_witness_sha256",
    "maximum_win_witness_sha256",
    "bound_type",
)
LEADER_HEADER = (
    "graph_id",
    "model_index",
    "model_id",
    "possible_leader",
    "necessary_leader",
    "classification",
    "certificate",
    "exhaustive_leader_count",
    "exhaustive_matching_count",
    "solver_status",
    "solver_maximin_decimal",
    "solver_maximin_upper_bound_decimal",
    "mip_gap_decimal",
    "witness_minimum_margin_numerator",
    "witness_minimum_margin_denominator",
    "witness_minimum_margin_decimal",
    "possible_witness_sha256",
)
HOLM_HEADER = (
    "graph_id",
    "comparison_index",
    "source_model_index",
    "source_model_id",
    "target_model_index",
    "target_model_id",
    "possible_holm_rejection",
    "necessary_holm_rejection",
    "classification",
    "certificate",
    "raw_p_best_numerator",
    "raw_p_best_denominator",
    "raw_p_best_decimal",
    "raw_p_worst_numerator",
    "raw_p_worst_denominator",
    "raw_p_worst_decimal",
    "witness_rejection_count",
    "witness_matching_count",
    "rejecting_witness_sha256",
    "nonrejecting_witness_sha256",
    "frequency_interpretation",
)
WITNESS_HEADER = (
    "witness_id",
    "graph_id",
    "objective_type",
    "objective_id",
    "sense",
    "membership_sha256",
    "selection_rank",
    "donor_id",
    "flu_year",
    "batch_id",
    "pool_id",
)


def methods_note(result: Mapping[str, Any]) -> str:
    lines = [
        "# Sound Life robust maximum-matching sensitivity V2",
        "",
        "**Evidence status:** Post hoc matching-sensitivity analysis conducted "
        "after inspection of the results.",
        "",
        "The donor–batch graphs, candidate edges and maximum cardinalities "
        "were reconstructed and hashed before the utility table was examined. "
        "Utilities were then used only to identify extremal matchings within "
        "the fixed graphs.",
        "",
        "The 12-wave training-batch-excluded graph was exhaustively enumerated "
        "over all 144 maximum matchings. For both graphs, additive minima and "
        "maxima were solved by exact rational min-cost maximum flow and were "
        "cross-checked against exhaustive enumeration where available.",
        "",
        "Possible leaders on the 39-wave graph were assessed with a fixed "
        "mixed-integer maximin formulation. Returned memberships were replayed "
        "against exact binary-rational utility values. Necessary-leader status "
        "uses exact lower bounds for all seven pairwise margins.",
        "",
        "Working Holm classifications on the 39-wave graph are deliberately "
        "conservative: only exhaustive results, sufficient Bonferroni/raw-p "
        "bounds, or explicit maximum-matching witnesses decide a claim. "
        "Undecided cases remain unresolved.",
        "",
        "The 256 salted matchings are retained only as deterministic witnesses. "
        "Their frequencies are not biological, sampling, or posterior "
        "probabilities.",
        "",
        "## Scope and limitations",
        "",
        "These results characterize sensitivity over declared maximum matchings "
        "of the recorded donor–batch graphs. They do not establish residual "
        "independence, identify a population-wide best configuration or validate "
        "the original protocol.",
        "",
        "## Graph summary",
        "",
        "| Graph | Maximum size | Analysis | Exhaustive count |",
        "|---|---:|---|---:|",
    ]
    for graph in result["graphs"]:
        exhaustive_count = graph["exhaustive_maximum_matching_count"]
        lines.append(
            f"| {graph['graph_id']} | {graph['maximum_cardinality']} | "
            f"{graph['analysis_mode']} | "
            f"{'' if exhaustive_count is None else exhaustive_count} |"
        )
    lines.append("")
    return "\n".join(lines)


def build(spec_path: Path, output_dir: Path) -> None:
    spec, spec_raw, project_root = load_spec(spec_path)
    output_dir = output_dir.resolve()
    require(
        not output_dir.exists() and not output_dir.is_symlink(),
        "output directory already exists",
    )
    bindings = input_bindings(spec)

    metadata_bindings = {
        role: verify_binding(project_root, bindings[role])
        for role in ("acquisition_metadata", "frozen_role_wave_mask")
    }
    candidates, train_batches, metadata_geometry = load_metadata(
        project_root, bindings
    )
    (
        design,
        graph_edges,
        exhaustive,
        salt_matchings,
        salts,
    ) = build_metadata_design(spec, candidates, train_batches)
    design_bytes = canonical_json_bytes(design)
    design_sha256 = sha256_bytes(design_bytes)

    utility_binding = verify_binding(
        project_root, bindings["frozen_eval_wave_utility"]
    )
    model_ids = tuple(spec["inference"]["model_ids_in_order"])
    utilities = load_utilities(
        project_root, bindings["frozen_eval_wave_utility"], model_ids
    )
    analysis = run_analysis(
        spec,
        design,
        graph_edges,
        exhaustive,
        salt_matchings,
        salts,
        utilities,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "SOUND_LIFE_MATCHING_ROBUSTNESS_RESULT_V2",
        "adapter": ADAPTER_ID,
        "status": "PASS_POST_HOC_ROBUST_MATCHING_SENSITIVITY_NO_CONFIRMATORY_AUTHORITY",
        "evidence_label": spec["evidence_label"],
        "design_sha256_frozen_before_utility_hash_or_open": design_sha256,
        "graphs": analysis["graph_results"],
        "inference": spec["inference"],
        "optimization": spec["optimization"],
        "authority_boundary": {
            **spec["authority_boundary"],
            "may_be_reported_as": (
                "POST_HOC_MAXIMUM_MATCHING_ROBUSTNESS_SENSITIVITY_ONLY"
            ),
            "population_generalizable": False,
        },
    }

    staging = output_dir.parent / f".{output_dir.name}.staging.{os.getpid()}"
    require(not staging.exists(), "staging directory already exists")
    staging.mkdir(parents=True)
    try:
        (staging / SPEC_SNAPSHOT).write_bytes(spec_raw)
        (staging / DESIGN_JSON).write_bytes(design_bytes)
        write_tsv(
            staging / EXHAUSTIVE_TSV,
            EXHAUSTIVE_HEADER,
            analysis["exhaustive_rows"],
        )
        write_tsv(staging / SALT_TSV, SALT_HEADER, analysis["salt_rows"])
        write_tsv(
            staging / INCLUSION_TSV,
            INCLUSION_HEADER,
            analysis["inclusion_rows"],
        )
        write_tsv(
            staging / UTILITY_TSV,
            UTILITY_OUTPUT_HEADER,
            analysis["utility_rows"],
        )
        write_tsv(
            staging / CONTRAST_TSV,
            CONTRAST_HEADER,
            analysis["contrast_rows"],
        )
        write_tsv(staging / WIN_TSV, WIN_HEADER, analysis["win_rows"])
        write_tsv(
            staging / LEADER_TSV, LEADER_HEADER, analysis["leader_rows"]
        )
        write_tsv(staging / HOLM_TSV, HOLM_HEADER, analysis["holm_rows"])
        write_tsv(
            staging / WITNESS_TSV,
            WITNESS_HEADER,
            analysis["witness_rows"],
        )
        (staging / RESULT_JSON).write_bytes(canonical_json_bytes(result))
        (staging / METHODS_MD).write_text(
            methods_note(result), encoding="utf-8", newline="\n"
        )

        receipt = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "SOUND_LIFE_MATCHING_ROBUSTNESS_RECEIPT_V2",
            "adapter": ADAPTER_ID,
            "status": "PASS",
            "spec": {
                "sha256": sha256_bytes(spec_raw),
                "snapshot": SPEC_SNAPSHOT,
            },
            "selection_firewall": {
                "ordered_stages": [
                    "VERIFY_METADATA_BINDINGS",
                    "LOAD_FROZEN_METADATA",
                    "CONSTRUCT_CANDIDATE_GRAPHS",
                    "FIX_MAXIMUM_CARDINALITIES",
                    "ENUMERATE_12_WAVE_GRAPH",
                    "FIX_256_SALT_WITNESS_MEMBERSHIPS",
                    "HASH_METADATA_DESIGN",
                    "VERIFY_UTILITY_BINDING",
                    "OPEN_FROZEN_UTILITY",
                    "RUN_OUTCOME_AWARE_ROBUST_OPTIMIZATION",
                ],
                "design_sha256_before_utility_hash_or_open": design_sha256,
                "utility_used_for_candidate_graph_or_cardinality": False,
                "utility_used_for_post_freeze_optimization_witnesses": True,
            },
            "input_bindings": {
                **metadata_bindings,
                "frozen_eval_wave_utility": utility_binding,
            },
            "implementation": implementation_receipt(),
            "geometry": {
                **metadata_geometry,
                "graph_count": len(graph_edges),
                "candidate_edge_rows": len(analysis["inclusion_rows"]),
                "exhaustive_membership_rows": len(
                    analysis["exhaustive_rows"]
                ),
                "salt_witness_membership_rows": len(analysis["salt_rows"]),
                "optimization_witness_rows": len(
                    analysis["witness_rows"]
                ),
                "utility_bound_rows": len(analysis["utility_rows"]),
                "contrast_bound_rows": len(analysis["contrast_rows"]),
                "win_bound_rows": len(analysis["win_rows"]),
                "leader_rows": len(analysis["leader_rows"]),
                "holm_rows": len(analysis["holm_rows"]),
            },
            "authority_boundary": result["authority_boundary"],
            "outputs_before_receipt": {
                name: {
                    "sha256": sha256_file(staging / name),
                    "bytes": (staging / name).stat().st_size,
                }
                for name in PAYLOAD_FILES
                if name != RECEIPT_JSON
            },
        }
        (staging / RECEIPT_JSON).write_bytes(canonical_json_bytes(receipt))
        (staging / MANIFEST).write_text(
            "".join(
                f"{sha256_file(staging / name)}  {name}\n"
                for name in PAYLOAD_FILES
            ),
            encoding="ascii",
            newline="\n",
        )
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verify(spec_path, output_dir)


def verify(spec_path: Path, bundle: Path) -> None:
    spec, spec_raw, _ = load_spec(spec_path)
    bundle = bundle.resolve()
    require(bundle.is_dir() and not bundle.is_symlink(), "bundle is absent or unsafe")
    entries = list(bundle.iterdir())
    expected = set(PAYLOAD_FILES) | {MANIFEST}
    require({path.name for path in entries} == expected, "bundle inventory differs")
    require(
        all(path.is_file() and not path.is_symlink() for path in entries),
        "unsafe bundle entry",
    )
    manifest_lines = (bundle / MANIFEST).read_text(
        encoding="ascii"
    ).splitlines()
    require(
        len(manifest_lines) == len(PAYLOAD_FILES),
        "manifest geometry differs",
    )
    for line, name in zip(manifest_lines, PAYLOAD_FILES):
        digest, separator, observed_name = line.partition("  ")
        require(
            separator == "  " and observed_name == name,
            "manifest order differs",
        )
        require(
            digest == sha256_file(bundle / name),
            f"manifest hash differs: {name}",
        )
    require(
        (bundle / SPEC_SNAPSHOT).read_bytes() == spec_raw,
        "spec snapshot differs",
    )
    design = json.loads((bundle / DESIGN_JSON).read_text(encoding="utf-8"))
    result = json.loads((bundle / RESULT_JSON).read_text(encoding="utf-8"))
    receipt = json.loads((bundle / RECEIPT_JSON).read_text(encoding="utf-8"))
    require(
        design["metadata_frozen_before_utility_hash_or_open"] is True
        and design["utility_used_to_construct_candidate_graphs"] is False,
        "metadata firewall differs",
    )
    require(
        receipt["selection_firewall"][
            "design_sha256_before_utility_hash_or_open"
        ]
        == sha256_file(bundle / DESIGN_JSON),
        "design binding differs",
    )
    require(
        result["status"].startswith("PASS_") and receipt["status"] == "PASS",
        "result/receipt status differs",
    )
    for key in (
        "confirmatory_authority",
        "residual_utility_independence_established",
        "frequency_is_natural_probability",
        "supports_population_best_configuration",
        "retroactively_validates_original_protocol",
        "population_generalizable",
    ):
        require(
            result["authority_boundary"][key] is False,
            f"scope record differs: {key}",
        )

    exhaustive_rows = read_tsv(bundle / EXHAUSTIVE_TSV, EXHAUSTIVE_HEADER)
    salt_rows = read_tsv(bundle / SALT_TSV, SALT_HEADER)
    inclusion_rows = read_tsv(bundle / INCLUSION_TSV, INCLUSION_HEADER)
    utility_rows = read_tsv(
        bundle / UTILITY_TSV, UTILITY_OUTPUT_HEADER
    )
    contrast_rows = read_tsv(bundle / CONTRAST_TSV, CONTRAST_HEADER)
    win_rows = read_tsv(bundle / WIN_TSV, WIN_HEADER)
    leader_rows = read_tsv(bundle / LEADER_TSV, LEADER_HEADER)
    holm_rows = read_tsv(bundle / HOLM_TSV, HOLM_HEADER)
    witness_rows = read_tsv(bundle / WITNESS_TSV, WITNESS_HEADER)

    graph_specs = {row["graph_id"]: row for row in spec["graphs"]}
    require(
        len(exhaustive_rows) == 144 * 12,
        "exhaustive membership geometry differs",
    )
    require(
        len(salt_rows)
        == spec["witness_ensemble"]["salt_count"]
        * sum(row["expected_maximum_cardinality"] for row in spec["graphs"]),
        "salt membership geometry differs",
    )
    require(
        len(inclusion_rows)
        == sum(row["expected_candidate_waves"] for row in spec["graphs"]),
        "inclusion-bound geometry differs",
    )
    require(len(utility_rows) == 2 * 8, "utility-bound geometry differs")
    require(len(contrast_rows) == 2 * 56, "contrast geometry differs")
    require(len(win_rows) == 2 * 56, "win-bound geometry differs")
    require(len(leader_rows) == 2 * 8, "leader geometry differs")
    require(len(holm_rows) == 2 * 56, "Holm geometry differs")

    exhaustive_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in exhaustive_rows:
        exhaustive_groups[(row["graph_id"], row["matching_index"])].append(row)
    require(len(exhaustive_groups) == 144, "exhaustive matching count differs")
    require(
        all(len(rows) == 12 for rows in exhaustive_groups.values()),
        "exhaustive matching cardinality differs",
    )
    require(
        len({rows[0]["membership_sha256"] for rows in exhaustive_groups.values()})
        == 144,
        "exhaustive memberships are not unique",
    )

    salt_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in salt_rows:
        salt_groups[(row["graph_id"], row["salt_index"])].append(row)
    require(
        len(salt_groups) == 2 * spec["witness_ensemble"]["salt_count"],
        "salt matching count differs",
    )
    for (graph_id, _), rows in salt_groups.items():
        require(
            len(rows)
            == graph_specs[graph_id]["expected_maximum_cardinality"],
            "salt matching cardinality differs",
        )

    witness_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in witness_rows:
        witness_groups[row["witness_id"]].append(row)
    require(witness_groups, "optimization witnesses are absent")
    for rows in witness_groups.values():
        graph_id = rows[0]["graph_id"]
        require(
            len(rows)
            == graph_specs[graph_id]["expected_maximum_cardinality"],
            "optimization witness cardinality differs",
        )
        require(
            len({row["membership_sha256"] for row in rows}) == 1,
            "optimization witness digest differs",
        )
        edges = [
            Edge(
                row["donor_id"],
                row["flu_year"],
                row["batch_id"],
                row["pool_id"],
                0,
                0,
            )
            for row in rows
        ]
        require(
            membership_digest(edges) == rows[0]["membership_sha256"],
            "optimization witness membership replay differs",
        )

    require(
        all(
            int(row["inclusion_min"])
            <= int(row["inclusion_max"])
            and row["optimization_scope"]
            == "ALL_MAXIMUM_CARDINALITY_MATCHINGS"
            for row in inclusion_rows
        ),
        "inclusion bounds differ",
    )
    require(
        all(
            row["bound_type"] == "EXACT_OVER_ALL_MAXIMUM_MATCHINGS"
            for row in utility_rows + contrast_rows + win_rows
        ),
        "exact-bound label differs",
    )
    require(
        all(
            row["classification"]
            in {
                "NECESSARY_LEADER",
                "POSSIBLE_NOT_NECESSARY_LEADER",
                "IMPOSSIBLE_LEADER",
                "UNRESOLVED_LEADER_POSSIBILITY",
            }
            for row in leader_rows
        ),
        "leader classification differs",
    )
    require(
        all(row["classification"] for row in holm_rows),
        "Holm classification absent",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sound-life-matching-robustness-v2"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--spec", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--spec", type=Path, required=True)
    verify_parser.add_argument("--bundle", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        build(args.spec, args.output_dir)
        payload = {
            "adapter": ADAPTER_ID,
            "status": "PASS",
            "bundle": str(args.output_dir),
        }
    else:
        verify(args.spec, args.bundle)
        payload = {
            "adapter": ADAPTER_ID,
            "status": "PASS_VERIFIED",
            "bundle": str(args.bundle),
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RobustnessError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)

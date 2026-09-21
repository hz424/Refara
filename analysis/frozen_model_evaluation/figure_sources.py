#!/usr/bin/env python3
"""Load and validate the deidentified source tables used by Figure 4 and ED Figure 7."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np


SOURCE_RELATIVE = Path("Frozen_model_evaluation_v1813")
FIGURE_RELATIVE = SOURCE_RELATIVE / "figure_source"
SUPPORT_RELATIVE = SOURCE_RELATIVE / "data" / "supporting_data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing figure source table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"Empty figure source table: {path}")
    return rows


def numeric(rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    values = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite value in figure source table")


def load_figure4(source_root: Path) -> dict[str, object]:
    root = source_root / FIGURE_RELATIVE
    paths = {
        "cross8_donors": root / "Figure_4d_cross8_donors.tsv",
        "cross8_summary": root / "Figure_4d_cross8_summary.tsv",
        "new39_batches": root / "Figure_4d_new39_batches.tsv",
        "new39_summary": root / "Figure_4d_new39_summary.tsv",
    }
    tables = {name: read_tsv(path) for name, path in paths.items()}
    cross = tables["cross8_donors"]
    cross_summary = tables["cross8_summary"]
    batches = tables["new39_batches"]
    new_summary = tables["new39_summary"]

    assert len(cross) == 16
    assert sorted({row["donor_id"] for row in cross}) == [f"H{i:02d}" for i in range(1, 9)]
    assert {row["evaluation_reference"] for row in cross} == {"shared_B1", "heldout_B2"}
    assert all(sum(row["donor_id"] == donor for row in cross) == 2
               for donor in {row["donor_id"] for row in cross})
    assert len(cross_summary) == 2
    assert {row["evaluation_reference"] for row in cross_summary} == {"shared_B1", "heldout_B2"}
    assert len(batches) == 8
    assert sorted(row["batch_id"] for row in batches) == [f"B{i:02d}" for i in range(1, 9)]
    assert len(new_summary) == 1 and new_summary[0]["evaluation_reference"] == "heldout_B2"

    metric_fields = ("mse_S", "mse_D", "mse_D_minus_S", "mae_S", "mae_D", "mae_D_minus_S")
    for rows in (cross, cross_summary, batches, new_summary):
        numeric(rows, metric_fields)
        for row in rows:
            assert np.isclose(float(row["mse_D"]) - float(row["mse_S"]),
                              float(row["mse_D_minus_S"]), atol=2e-15, rtol=0)
            assert np.isclose(float(row["mae_D"]) - float(row["mae_S"]),
                              float(row["mae_D_minus_S"]), atol=2e-15, rtol=0)

    for reference in ("shared_B1", "heldout_B2"):
        donor_rows = [row for row in cross if row["evaluation_reference"] == reference]
        summary = next(row for row in cross_summary if row["evaluation_reference"] == reference)
        for field in metric_fields:
            assert np.isclose(np.mean([float(row[field]) for row in donor_rows]),
                              float(summary[field]), atol=2e-15, rtol=0)
    for field in metric_fields:
        assert np.isclose(np.mean([float(row[field]) for row in batches]),
                          float(new_summary[0][field]), atol=2e-15, rtol=0)

    assert float(next(row for row in cross_summary if row["evaluation_reference"] == "shared_B1")
                 ["mse_D_minus_S"]) > 0
    assert float(next(row for row in cross_summary if row["evaluation_reference"] == "heldout_B2")
                 ["mse_D_minus_S"]) < 0
    assert all(float(row["mse_D_minus_S"]) < 0 for row in batches)
    assert int(new_summary[0]["donor_count"]) == 39
    assert int(new_summary[0]["task_count"]) == 78

    return {
        **tables,
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in paths.values()},
    }


def load_ed7(source_root: Path) -> dict[str, object]:
    root = source_root / SUPPORT_RELATIVE
    paths = {
        "cross8_assignments": root / "cross8_assignment_sensitivity.tsv",
        "new39_assignments": root / "new39_assignment_sensitivity.tsv",
        "donors": root / "new39_primary_donor_metrics.tsv",
        "orientations": root / "new39_primary_orientation_metrics.tsv",
        "extensions": root / "new39_primary_summary_by_cohort_panel_weight.tsv",
        "leave_one_batch_out": root / "new39_primary_leave_one_batch_out.tsv",
        "programs": root / "new39_primary_program_contributions.tsv",
    }
    tables = {name: read_tsv(path) for name, path in paths.items()}

    cross = tables["cross8_assignments"]
    assert len(cross) == 90
    assert sum(row["evaluation"] == "shared_observation" for row in cross) == 30
    assert sum(row["evaluation"] == "independent_observation" for row in cross) == 60

    new_assign = [row for row in tables["new39_assignments"]
                  if row["cohort"] == "primary39_two_types"]
    assert len(new_assign) == 120
    assert sum(row["evaluation"] == "shared_observation" for row in new_assign) == 30
    assert sum(row["evaluation"] == "independent_observation" for row in new_assign) == 60
    assert sum(row["evaluation"] == "deeper_observation" for row in new_assign) == 30

    donors = [row for row in tables["donors"] if row["cohort"] == "primary39_two_types"]
    orientations = [row for row in tables["orientations"] if row["cohort"] == "primary39_two_types"]
    extensions = tables["extensions"]
    loo = [row for row in tables["leave_one_batch_out"] if row["cohort"] == "primary39_two_types"]
    programs = [row for row in tables["programs"] if row["cohort"] == "primary39_two_types"]
    assert len(donors) == 39 and len({row["donor_id"] for row in donors}) == 39
    assert {row["orientation"] for row in donors} == {"O1", "O2"}
    assert len(orientations) == 2 and {row["orientation"] for row in orientations} == {"O1", "O2"}
    assert len(extensions) == 18
    assert len(loo) == 8 and len({row["omitted_batch_id"] for row in loo}) == 8
    assert len(programs) == 37 and len({row["program_id"] for row in programs}) == 37

    for rows, field in ((cross, "program_mse_D_minus_S"), (new_assign, "mse_D_minus_S"),
                        (donors, "mse_D_minus_S"), (orientations, "mse_D_minus_S"),
                        (extensions, "mse_D_minus_S"), (loo, "mse_D_minus_S"),
                        (programs, "mse_D_minus_S")):
        numeric(rows, (field,))

    assert all(float(row["program_mse_D_minus_S"]) > 0
               for row in cross if row["evaluation"] == "shared_observation")
    assert all(float(row["program_mse_D_minus_S"]) < 0
               for row in cross if row["evaluation"] == "independent_observation")
    assert all(float(row["mse_D_minus_S"]) > 0
               for row in new_assign if row["evaluation"] == "shared_observation")
    assert all(float(row["mse_D_minus_S"]) < 0
               for row in new_assign if row["evaluation"] != "shared_observation")
    assert all(float(row["mse_D_minus_S"]) < 0 for row in extensions)
    assert all(float(row["mse_D_minus_S"]) < 0 for row in loo)
    assert sum(float(row["mse_D_minus_S"]) < 0 for row in programs) == 35

    return {
        "cross8_assignments": cross,
        "new39_assignments": new_assign,
        "donors": donors,
        "orientations": orientations,
        "extensions": extensions,
        "leave_one_batch_out": loo,
        "programs": programs,
        "source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in paths.values()},
    }

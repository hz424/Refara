from __future__ import annotations

import csv
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "benchmark_design_audit.py"
CONFIG = ROOT / "example" / "config.json"
EXPECTED_TREE = {
    "BENCHMARK_DESIGN_AUDIT_MANIFEST_V1.sha256",
    "CONFIG_SNAPSHOT.json",
    "KMIN_RESOLUTION_V1.tsv",
    "PAIRWISE_FAMILY_PLAN_V1.tsv",
    "PLAN_RECEIPT_V1.json",
    "REFERENCE_ROLE_ALLOCATIONS_V1.tsv",
    "REPORTING_CHECKLIST_V1.md",
    "ROLE_OVERLAP_MATRIX_V1.tsv",
}


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-s", "-B", str(CLI), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class PlanComparisonsCliTests(unittest.TestCase):
    def test_plan_emits_explicit_role_geometry_complete_ordered_family_and_kmin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_plan_") as temporary:
            output = Path(temporary) / "bundle"
            completed = run(
                "plan-comparisons",
                "--config",
                str(CONFIG),
                "--output-dir",
                str(output),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual({path.name for path in output.iterdir()}, EXPECTED_TREE)
            allocations = rows(output / "REFERENCE_ROLE_ALLOCATIONS_V1.tsv")
            overlap = rows(output / "ROLE_OVERLAP_MATRIX_V1.tsv")
            family = rows(output / "PAIRWISE_FAMILY_PLAN_V1.tsv")
            resolution = rows(output / "KMIN_RESOLUTION_V1.tsv")
            self.assertEqual(len(allocations), 5)
            self.assertEqual(
                {row["aggregation_order"] for row in allocations},
                {"SCORE_WITHIN_ROTATION_THEN_EQUAL_WEIGHT_ROTATIONS"},
            )
            self.assertEqual(len(overlap), 45)
            shared = [row for row in overlap if row["allocation_id"] == "shared_D"]
            self.assertEqual({row["shares_control_block"] for row in shared}, {"1"})
            disjoint = [row for row in overlap if row["allocation_id"] == "disjoint_D"]
            self.assertEqual(sum(row["shares_control_block"] == "1" for row in disjoint), 3)
            self.assertEqual(len(family), 18)
            expected_pairs = set(itertools.permutations(("METHOD_A", "METHOD_B", "METHOD_C"), 2))
            for allocation_id in ("shared_D", "disjoint_D", "crossfit_D"):
                observed = {
                    (row["model_a_id"], row["model_b_id"])
                    for row in family
                    if row["allocation_id"] == allocation_id
                }
                self.assertEqual(observed, expected_pairs)
            self.assertEqual(len(resolution), 3)
            self.assertEqual({row["complete_family_size"] for row in resolution}, {"6"})
            self.assertEqual({row["k_min"] for row in resolution}, {"7"})
            self.assertEqual(
                {row["interpretation"] for row in resolution},
                {"ATTAINABILITY_ONLY_NOT_AN_EFFECT_OR_INDEPENDENCE_CLAIM"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

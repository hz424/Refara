from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "benchmark_design_audit.py"
CONFIG = ROOT / "example" / "config.json"
METADATA = ROOT / "example" / "metadata.tsv"
UTILITIES = ROOT / "example" / "utilities.tsv"
EXPECTED_TREE = {
    "BENCHMARK_DESIGN_AUDIT_MANIFEST_V1.sha256",
    "CONFIG_SNAPSHOT.json",
    "DESCRIPTIVE_SUMMARIES_V1.tsv",
    "EVALUATION_RECEIPT_V1.json",
    "INPUT_UNIT_METADATA.tsv",
    "INPUT_UTILITIES.tsv",
    "PAIRWISE_COMPLETE_FAMILY_V1.tsv",
    "REPORTING_CHECKLIST_V1.md",
    "UNIT_AUDIT_MANIFEST_SNAPSHOT.sha256",
    "UNIT_AUDIT_RECEIPT_SNAPSHOT.json",
    "UNIT_BY_METHOD_UTILITY_V1.tsv",
    "WITHHELD_COMPARISONS_V1.tsv",
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


class EvaluateCliTests(unittest.TestCase):
    def test_evaluate_emits_exact_complete_family_holm_and_canonical_decisions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_evaluate_") as temporary:
            root = Path(temporary)
            unit_audit = root / "unit-audit"
            audited = run(
                "audit-units",
                "--config",
                str(CONFIG),
                "--metadata",
                str(METADATA),
                "--output-dir",
                str(unit_audit),
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            output = root / "bundle"
            completed = run(
                "evaluate",
                "--config",
                str(CONFIG),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(unit_audit),
                "--output-dir",
                str(output),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual({path.name for path in output.iterdir()}, EXPECTED_TREE)
            unit_utilities = rows(output / "UNIT_BY_METHOD_UTILITY_V1.tsv")
            pairwise = rows(output / "PAIRWISE_COMPLETE_FAMILY_V1.tsv")
            summaries = rows(output / "DESCRIPTIVE_SUMMARIES_V1.tsv")
            self.assertEqual(len(unit_utilities), 72)
            self.assertEqual(len(pairwise), 18)
            self.assertEqual(len(summaries), 9)
            self.assertEqual(
                {row["decision"] for row in pairwise},
                {"DIRECTION", "NO_DIRECTION", "WITHHELD"},
            )

            shared = next(
                row for row in pairwise
                if row["allocation_id"] == "shared_D"
                and row["model_a_id"] == "METHOD_A"
                and row["model_b_id"] == "METHOD_B"
            )
            self.assertEqual(shared["wins"], "8")
            self.assertEqual(shared["losses"], "0")
            self.assertEqual(shared["literal_zero_ties"], "0")
            self.assertEqual(shared["exact_one_sided_p"], "0.00390625")
            self.assertEqual(shared["holm_adjusted_p"], "0.0234375")
            self.assertEqual(shared["reject_at_alpha"], "1")
            self.assertEqual(shared["decision"], "DIRECTION")
            self.assertEqual(shared["withheld_reason"], "")

            tied = next(
                row for row in pairwise
                if row["allocation_id"] == "disjoint_D"
                and row["model_a_id"] == "METHOD_A"
                and row["model_b_id"] == "METHOD_B"
            )
            self.assertEqual(tied["wins"], "7")
            self.assertEqual(tied["losses"], "0")
            self.assertEqual(tied["literal_zero_ties"], "1")
            self.assertEqual(tied["effective_non_tie_units"], "7")
            self.assertEqual(tied["exact_one_sided_p"], "0.0078125")

            below_kmin = [
                row
                for row in pairwise
                if row["withheld_reason"] == "ARITHMETIC_SUPPORT_BELOW_KMIN"
            ]
            self.assertEqual(len(below_kmin), 6)
            self.assertEqual({row["allocation_id"] for row in below_kmin}, {"crossfit_D"})
            self.assertEqual({row["effective_non_tie_units"] for row in below_kmin}, {"0"})
            self.assertEqual({row["literal_zero_ties"] for row in below_kmin}, {"8"})
            self.assertEqual({row["decision"] for row in below_kmin}, {"WITHHELD"})
            self.assertEqual(
                {row["withheld_reason"] for row in below_kmin},
                {"ARITHMETIC_SUPPORT_BELOW_KMIN"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

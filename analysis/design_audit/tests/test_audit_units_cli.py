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
EXPECTED_TREE = {
    "ACQUISITION_EDGE_CANDIDATES_V1.tsv",
    "AUDIT_RECEIPT_V1.json",
    "BENCHMARK_DESIGN_AUDIT_MANIFEST_V1.sha256",
    "CONFIG_SNAPSHOT.json",
    "INPUT_METADATA.tsv",
    "REPORTING_CHECKLIST_V1.md",
    "UNIT_AUDIT_V1.tsv",
    "UNIT_GROUP_GEOMETRY_V1.tsv",
    "UNIT_HIERARCHY_V1.tsv",
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


class AuditUnitsCliTests(unittest.TestCase):
    def test_audit_reports_geometry_and_records_but_does_not_verify_independence_assertion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_units_") as temporary:
            output = Path(temporary) / "bundle"
            completed = run(
                "audit-units",
                "--config",
                str(CONFIG),
                "--metadata",
                str(METADATA),
                "--output-dir",
                str(output),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual({path.name for path in output.iterdir()}, EXPECTED_TREE)
            hierarchy = rows(output / "UNIT_HIERARCHY_V1.tsv")
            units = rows(output / "UNIT_AUDIT_V1.tsv")
            geometry = rows(output / "UNIT_GROUP_GEOMETRY_V1.tsv")
            edges = rows(output / "ACQUISITION_EDGE_CANDIDATES_V1.tsv")
            self.assertEqual(len(hierarchy), 16)
            self.assertEqual(len(units), 8)
            self.assertTrue(all(row["object_count"] == "2" for row in units))
            self.assertTrue(all(row["repeated_objects_beyond_first"] == "1" for row in units))
            self.assertEqual(
                {row["independence_status"] for row in units},
                {"ESTABLISHED_BY_EXPERT_ASSERTION"},
            )
            self.assertEqual(len(geometry), 10)
            self.assertEqual(len(edges), 20)
            self.assertEqual(
                {row["interpretation"] for row in geometry + edges},
                {"CANDIDATE_DEPENDENCE_GEOMETRY_NOT_INDEPENDENCE_EVIDENCE"},
            )
            receipt = (output / "AUDIT_RECEIPT_V1.json").read_text(encoding="utf-8")
            self.assertIn('"independence_assertion_machine_verified":false', receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "public_empirical_example" / "run_public_empirical_example.py"
SOURCE_DATA_ROOT = os.environ.get("REFERENCE_CELL_SOURCE_DATA_ROOT")


class PublicEmpiricalExampleTests(unittest.TestCase):
    @unittest.skipUnless(SOURCE_DATA_ROOT, "external Source Data root was not supplied")
    def test_released_root_utilities_run_as_downstream_support_audit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="public_empirical_design_audit_") as temporary:
            output = Path(temporary) / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-s",
                    "-B",
                    str(RUNNER),
                    "--output-root",
                    str(output),
                    "--source-data-root",
                    str(SOURCE_DATA_ROOT),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("PASS_UTILITY_LEVEL_SUPPORT_AUDIT", completed.stdout)
            receipt = json.loads(
                (output / "evaluation" / "EVALUATION_RECEIPT_V1.json").read_text()
            )
            self.assertEqual(receipt["unit_utility_count"], 192)
            self.assertEqual(receipt["pairwise_comparison_count"], 168)
            self.assertEqual(receipt["directional_rejection_count"], 0)
            self.assertEqual(receipt["withheld_comparison_count"], 168)
            with (output / "evaluation" / "PAIRWISE_COMPLETE_FAMILY_V1.tsv").open(
                newline="", encoding="utf-8"
            ) as stream:
                pairwise = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual({row["decision"] for row in pairwise}, {"WITHHELD"})
            self.assertEqual(
                {row["withheld_reason"] for row in pairwise},
                {"ARITHMETIC_SUPPORT_BELOW_KMIN"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

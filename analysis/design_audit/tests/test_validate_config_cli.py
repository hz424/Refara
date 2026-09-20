from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "benchmark_design_audit.py"
CONFIG = ROOT / "example" / "config.json"


class ValidateConfigCliTests(unittest.TestCase):
    def test_valid_config_reports_registered_contract_and_resolution(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-s",
                "-B",
                str(CLI),
                "validate-config",
                "--config",
                str(CONFIG),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "PASS_DECLARATIVE_CONTRACT")
        self.assertEqual(report["audit_id"], "SYNTHETIC_REFERENCE_ALLOCATION_EXAMPLE_V1")
        self.assertEqual(report["complete_ordered_comparisons_per_family"], 6)
        self.assertEqual(report["k_min_for_smallest_holm_threshold"], 7)
        self.assertEqual(report["tie_rule"], "LITERAL_IEEE754_BINARY64_ZERO")
        self.assertTrue(report["declared_unit_independence_established"])
        self.assertFalse(report["upstream_provenance_machine_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

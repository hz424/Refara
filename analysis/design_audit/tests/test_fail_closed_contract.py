from __future__ import annotations

import csv
import json
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


class FailClosedContractTests(unittest.TestCase):
    def write_config(self, directory: Path, name: str, value: dict) -> Path:
        path = directory / f"{name}.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def test_missing_claimed_fields_and_malformed_role_geometry_fail_validation(self) -> None:
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        cases = {}

        missing_independence = json.loads(json.dumps(base))
        missing_independence["unit_contract"].pop("independence_justification")
        cases["missing_independence"] = missing_independence

        missing_output_semantics = json.loads(json.dumps(base))
        missing_output_semantics["configurations"][0].pop("output_semantics")
        cases["missing_output_semantics"] = missing_output_semantics

        missing_terminal_identity = json.loads(json.dumps(base))
        missing_terminal_identity["configurations"][0].pop("terminal_state_identity")
        cases["missing_terminal_identity"] = missing_terminal_identity

        malformed_role_mapping = json.loads(json.dumps(base))
        malformed_role_mapping["allocations"][0]["rotations"][0]["observed_effect"] = []
        cases["malformed_role_mapping"] = malformed_role_mapping

        unintended_overlap = json.loads(json.dumps(base))
        unintended_overlap["allocations"][1]["rotations"][0]["prediction_reference"] = "B1"
        cases["unintended_overlap"] = unintended_overlap

        missing_depth = json.loads(json.dumps(base))
        missing_depth["allocations"][0]["rotations"][0]["role_depth_labels"].pop(
            "observed_effect"
        )
        cases["missing_depth"] = missing_depth

        missing_aggregation = json.loads(json.dumps(base))
        missing_aggregation["allocations"][0].pop("aggregation_order")
        cases["missing_aggregation"] = missing_aggregation

        missing_timing = json.loads(json.dumps(base))
        missing_timing.pop("analysis_timing")
        cases["missing_timing"] = missing_timing

        with tempfile.TemporaryDirectory(prefix="design_audit_invalid_contract_") as temporary:
            root = Path(temporary)
            for name, value in cases.items():
                with self.subTest(case=name):
                    path = self.write_config(root, name, value)
                    completed = run("validate-config", "--config", str(path))
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("BENCHMARK_DESIGN_AUDIT_ERROR", completed.stderr)

    def test_false_independence_emits_withheld_only_and_verifies(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        justification = config["unit_contract"]["independence_justification"]
        justification["status"] = "NOT_ESTABLISHED"
        justification["basis"] = "SYNTHETIC_NEGATIVE_TEST_WITH_NO_INDEPENDENCE_JUSTIFICATION"
        with tempfile.TemporaryDirectory(prefix="design_audit_false_independence_") as temporary:
            root = Path(temporary)
            config_path = self.write_config(root, "config", config)
            unit_audit = root / "unit-audit"
            audited = run(
                "audit-units",
                "--config",
                str(config_path),
                "--metadata",
                str(METADATA),
                "--output-dir",
                str(unit_audit),
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            evaluation = root / "evaluation"
            evaluated = run(
                "evaluate",
                "--config",
                str(config_path),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(unit_audit),
                "--output-dir",
                str(evaluation),
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            pairwise = rows(evaluation / "PAIRWISE_COMPLETE_FAMILY_V1.tsv")
            self.assertEqual({row["decision"] for row in pairwise}, {"WITHHELD"})
            self.assertEqual(
                {row["withheld_reason"] for row in pairwise},
                {"UNIT_INDEPENDENCE_NOT_ESTABLISHED"},
            )
            self.assertEqual({row["reject_at_alpha"] for row in pairwise}, {"NA"})
            receipt = json.loads((evaluation / "EVALUATION_RECEIPT_V1.json").read_text())
            self.assertEqual(receipt["directional_rejection_count"], 0)
            self.assertEqual(receipt["status"], "WITHHELD_REQUIRED_GATE_FAILURE")
            verified = run(
                "verify-output",
                "--config",
                str(config_path),
                "--bundle",
                str(evaluation),
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_false_training_evaluation_separation_emits_withheld_only_and_verifies(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        separation = config["configurations"][0]["training_evaluation_separation"]
        separation["status"] = "NOT_ESTABLISHED"
        separation["basis"] = "SYNTHETIC_NEGATIVE_TEST_WITH_NO_SEPARATION_JUSTIFICATION"
        with tempfile.TemporaryDirectory(prefix="design_audit_false_separation_") as temporary:
            root = Path(temporary)
            config_path = self.write_config(root, "config", config)
            unit_audit = root / "unit-audit"
            audited = run(
                "audit-units",
                "--config",
                str(config_path),
                "--metadata",
                str(METADATA),
                "--output-dir",
                str(unit_audit),
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            evaluation = root / "evaluation"
            evaluated = run(
                "evaluate",
                "--config",
                str(config_path),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(unit_audit),
                "--output-dir",
                str(evaluation),
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            pairwise = rows(evaluation / "PAIRWISE_COMPLETE_FAMILY_V1.tsv")
            self.assertEqual({row["decision"] for row in pairwise}, {"WITHHELD"})
            self.assertEqual(
                {row["withheld_reason"] for row in pairwise},
                {"TRAINING_EVALUATION_SEPARATION_NOT_ESTABLISHED"},
            )
            self.assertEqual({row["reject_at_alpha"] for row in pairwise}, {"NA"})
            receipt = json.loads((evaluation / "EVALUATION_RECEIPT_V1.json").read_text())
            self.assertEqual(receipt["directional_rejection_count"], 0)
            self.assertEqual(receipt["status"], "WITHHELD_REQUIRED_GATE_FAILURE")
            self.assertEqual(
                receipt["gate_failures"],
                ["TRAINING_EVALUATION_SEPARATION_NOT_ESTABLISHED"],
            )
            verified = run(
                "verify-output",
                "--config",
                str(config_path),
                "--bundle",
                str(evaluation),
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_altered_configuration_identity_cannot_reuse_prior_unit_audit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_changed_identity_") as temporary:
            root = Path(temporary)
            unit_audit = root / "unit-audit"
            self.assertEqual(
                run(
                    "audit-units",
                    "--config",
                    str(CONFIG),
                    "--metadata",
                    str(METADATA),
                    "--output-dir",
                    str(unit_audit),
                ).returncode,
                0,
            )
            changed = json.loads(CONFIG.read_text(encoding="utf-8"))
            changed["configurations"][0]["implementation_identity"] = (
                "ALTERED_IMPLEMENTATION_IDENTITY"
            )
            changed_path = self.write_config(root, "changed", changed)
            output = root / "evaluation"
            completed = run(
                "evaluate",
                "--config",
                str(changed_path),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(unit_audit),
                "--output-dir",
                str(output),
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

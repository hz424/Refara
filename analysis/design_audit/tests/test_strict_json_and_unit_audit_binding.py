from __future__ import annotations

import hashlib
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
MANIFEST = "BENCHMARK_DESIGN_AUDIT_MANIFEST_V1.sha256"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-s", "-B", str(CLI), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def rehash_manifest_entry(bundle: Path, name: str) -> None:
    changed_digest = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    manifest = bundle / MANIFEST
    changed_lines = []
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, declared_name = line.split("  ", 1)
        if declared_name == name:
            digest = changed_digest
        changed_lines.append(f"{digest}  {declared_name}")
    manifest.write_text("\n".join(changed_lines) + "\n", encoding="ascii")


def inject_duplicate_json_key(path: Path, key: str, first_value: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f'"{key}":'
    position = text.index(marker)
    duplicate = f'"{key}":{json.dumps(first_value, separators=(",", ":"))},'
    path.write_text(text[:position] + duplicate + text[position:], encoding="utf-8")


class StrictJsonAndUnitAuditBindingTests(unittest.TestCase):
    def generate_audit(self, root: Path, name: str = "unit-audit") -> Path:
        output = root / name
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
        return output

    def generate_evaluation(self, root: Path, audit: Path, name: str = "evaluation") -> Path:
        output = root / name
        completed = run(
            "evaluate",
            "--config",
            str(CONFIG),
            "--utilities",
            str(UTILITIES),
            "--unit-audit-bundle",
            str(audit),
            "--output-dir",
            str(output),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return output

    def test_duplicate_config_keys_fail_validate_audit_evaluate_and_verify(self) -> None:
        source = CONFIG.read_text(encoding="utf-8")
        independence_marker = (
            '"independence_justification": {\n'
            '      "status": "ESTABLISHED_BY_EXPERT_ASSERTION",'
        )
        duplicate_independence = source.replace(
            independence_marker,
            '"independence_justification": {\n'
            '      "status": "NOT_ESTABLISHED",\n'
            '      "status": "ESTABLISHED_BY_EXPERT_ASSERTION",',
            1,
        )
        self.assertNotEqual(source, duplicate_independence)
        duplicate_top_level = source.replace(
            '{\n  "schema_version": "1.0.0",',
            '{\n  "audit_id": "FIRST_DUPLICATE_AUDIT_ID",\n'
            '  "schema_version": "1.0.0",',
            1,
        )
        self.assertNotEqual(source, duplicate_top_level)

        with tempfile.TemporaryDirectory(prefix="design_audit_duplicate_config_") as temporary:
            root = Path(temporary)
            nested_path = root / "duplicate-independence.json"
            nested_path.write_text(duplicate_independence, encoding="utf-8")
            top_path = root / "duplicate-top-level.json"
            top_path.write_text(duplicate_top_level, encoding="utf-8")
            audit = self.generate_audit(root)

            commands = (
                ("validate", ("validate-config", "--config", str(nested_path)), None),
                (
                    "validate-top-level",
                    ("validate-config", "--config", str(top_path)),
                    None,
                ),
                (
                    "audit",
                    (
                        "audit-units",
                        "--config",
                        str(nested_path),
                        "--metadata",
                        str(METADATA),
                        "--output-dir",
                        str(root / "invalid-audit"),
                    ),
                    root / "invalid-audit",
                ),
                (
                    "evaluate",
                    (
                        "evaluate",
                        "--config",
                        str(nested_path),
                        "--utilities",
                        str(UTILITIES),
                        "--unit-audit-bundle",
                        str(audit),
                        "--output-dir",
                        str(root / "invalid-evaluation"),
                    ),
                    root / "invalid-evaluation",
                ),
                (
                    "verify",
                    (
                        "verify-output",
                        "--config",
                        str(top_path),
                        "--bundle",
                        str(audit),
                    ),
                    None,
                ),
            )
            for name, arguments, output in commands:
                with self.subTest(command=name):
                    completed = run(*arguments)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn("duplicate JSON object key", completed.stderr)
                    if output is not None:
                        self.assertFalse(output.exists())

            valid = run("validate-config", "--config", str(CONFIG))
            self.assertEqual(valid.returncode, 0, valid.stderr)

    def test_duplicate_bundle_json_keys_and_manifest_names_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_duplicate_bundle_") as temporary:
            root = Path(temporary)

            receipt_audit = self.generate_audit(root, "receipt-audit")
            receipt_path = receipt_audit / "AUDIT_RECEIPT_V1.json"
            inject_duplicate_json_key(receipt_path, "status", "WITHHELD_REQUIRED_GATE_FAILURE")
            rehash_manifest_entry(receipt_audit, receipt_path.name)
            evaluated = run(
                "evaluate",
                "--config",
                str(CONFIG),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(receipt_audit),
                "--output-dir",
                str(root / "receipt-evaluation"),
            )
            self.assertNotEqual(evaluated.returncode, 0)
            self.assertIn("duplicate JSON object key: status", evaluated.stderr)
            verified = run(
                "verify-output",
                "--config",
                str(CONFIG),
                "--bundle",
                str(receipt_audit),
            )
            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("duplicate JSON object key: status", verified.stderr)

            snapshot_audit = self.generate_audit(root, "snapshot-audit")
            snapshot_path = snapshot_audit / "CONFIG_SNAPSHOT.json"
            inject_duplicate_json_key(snapshot_path, "audit_id", "FIRST_DUPLICATE_AUDIT_ID")
            rehash_manifest_entry(snapshot_audit, snapshot_path.name)
            verified = run(
                "verify-output",
                "--config",
                str(CONFIG),
                "--bundle",
                str(snapshot_audit),
            )
            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("duplicate JSON object key: audit_id", verified.stderr)

            manifest_audit = self.generate_audit(root, "manifest-audit")
            manifest_path = manifest_audit / MANIFEST
            manifest_lines = manifest_path.read_text(encoding="ascii").splitlines()
            manifest_path.write_text(
                "\n".join([*manifest_lines, manifest_lines[0]]) + "\n",
                encoding="ascii",
            )
            evaluated = run(
                "evaluate",
                "--config",
                str(CONFIG),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(manifest_audit),
                "--output-dir",
                str(root / "manifest-evaluation"),
            )
            self.assertNotEqual(evaluated.returncode, 0)
            verified = run(
                "verify-output",
                "--config",
                str(CONFIG),
                "--bundle",
                str(manifest_audit),
            )
            self.assertNotEqual(verified.returncode, 0)

            valid_audit = self.generate_audit(root, "valid-audit")
            evaluation = self.generate_evaluation(root, valid_audit)
            evaluation_receipt = evaluation / "EVALUATION_RECEIPT_V1.json"
            inject_duplicate_json_key(
                evaluation_receipt,
                "status",
                "WITHHELD_REQUIRED_GATE_FAILURE",
            )
            rehash_manifest_entry(evaluation, evaluation_receipt.name)
            verified = run(
                "verify-output",
                "--config",
                str(CONFIG),
                "--bundle",
                str(evaluation),
            )
            self.assertNotEqual(verified.returncode, 0)
            self.assertIn("duplicate JSON object key: status", verified.stderr)

    def test_rehashed_edge_tamper_fails_evaluate_and_portable_evaluation_verifier(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_edge_tamper_") as temporary:
            root = Path(temporary)
            audit = self.generate_audit(root)
            evaluation = self.generate_evaluation(root, audit)

            edge_path = audit / "ACQUISITION_EDGE_CANDIDATES_V1.tsv"
            original = edge_path.read_text(encoding="utf-8")
            changed = original.replace("\tU01\tU02\t", "\tU01\tU08\t", 1)
            self.assertNotEqual(original, changed)
            edge_path.write_text(changed, encoding="utf-8")
            rehash_manifest_entry(audit, edge_path.name)

            rejected_output = root / "rejected-evaluation"
            evaluated = run(
                "evaluate",
                "--config",
                str(CONFIG),
                "--utilities",
                str(UTILITIES),
                "--unit-audit-bundle",
                str(audit),
                "--output-dir",
                str(rejected_output),
            )
            self.assertNotEqual(evaluated.returncode, 0)
            self.assertFalse(rejected_output.exists())
            self.assertIn("unit-audit edges differ", evaluated.stderr)

            manifest_snapshot = evaluation / "UNIT_AUDIT_MANIFEST_SNAPSHOT.sha256"
            manifest_snapshot.write_bytes((audit / MANIFEST).read_bytes())
            receipt_path = evaluation / "EVALUATION_RECEIPT_V1.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["unit_audit_manifest_sha256"] = hashlib.sha256(
                manifest_snapshot.read_bytes()
            ).hexdigest()
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            rehash_manifest_entry(evaluation, manifest_snapshot.name)
            rehash_manifest_entry(evaluation, receipt_path.name)
            verified = run(
                "verify-output",
                "--config",
                str(CONFIG),
                "--bundle",
                str(evaluation),
            )
            self.assertNotEqual(verified.returncode, 0)
            self.assertIn(
                "unit-audit manifest snapshot does not bind complete verified semantics",
                verified.stderr,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

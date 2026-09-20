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


class VerifyOutputCliTests(unittest.TestCase):
    def test_verify_accepts_each_newly_generated_bundle_kind(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_verify_") as temporary:
            root = Path(temporary)
            commands = (
                ("audit", ("audit-units", "--config", str(CONFIG), "--metadata", str(METADATA))),
                ("plan", ("plan-comparisons", "--config", str(CONFIG))),
                (
                    "evaluation",
                    (
                        "evaluate",
                        "--config",
                        str(CONFIG),
                        "--utilities",
                        str(UTILITIES),
                        "--unit-audit-bundle",
                        str(root / "audit"),
                    ),
                ),
            )
            for name, arguments in commands:
                with self.subTest(bundle=name):
                    output = root / name
                    generated = run(*arguments, "--output-dir", str(output))
                    self.assertEqual(generated.returncode, 0, generated.stderr)
                    verified = run(
                        "verify-output",
                        "--config",
                        str(CONFIG),
                        "--bundle",
                        str(output),
                    )
                    self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_verify_rejects_semantic_tamper_even_after_manifest_is_rehashed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_tamper_") as temporary:
            output = Path(temporary) / "plan"
            generated = run(
                "plan-comparisons", "--config", str(CONFIG), "--output-dir", str(output)
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            matrix = output / "ROLE_OVERLAP_MATRIX_V1.tsv"
            original = matrix.read_text(encoding="utf-8")
            self.assertIn("\t1\n", original)
            matrix.write_text(original.replace("\t1\n", "\t0\n", 1), encoding="utf-8")
            changed_digest = hashlib.sha256(matrix.read_bytes()).hexdigest()
            manifest = output / MANIFEST
            lines = []
            for line in manifest.read_text(encoding="ascii").splitlines():
                digest, name = line.split("  ", 1)
                if name == matrix.name:
                    digest = changed_digest
                lines.append(f"{digest}  {name}")
            manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
            verified = run(
                "verify-output", "--config", str(CONFIG), "--bundle", str(output)
            )
            self.assertNotEqual(verified.returncode, 0)

    def test_verify_rejects_valid_but_substituted_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_config_substitution_") as temporary:
            root = Path(temporary)
            output = root / "plan"
            self.assertEqual(
                run("plan-comparisons", "--config", str(CONFIG), "--output-dir", str(output)).returncode,
                0,
            )
            changed = json.loads(CONFIG.read_text(encoding="utf-8"))
            changed["audit_id"] = "SUBSTITUTED_VALID_CONFIG"
            substitute = root / "substitute.json"
            substitute.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
            verified = run(
                "verify-output", "--config", str(substitute), "--bundle", str(output)
            )
            self.assertNotEqual(verified.returncode, 0)

    def test_evaluate_rejects_missing_or_duplicate_support_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_support_gate_") as temporary:
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
            source_lines = UTILITIES.read_text(encoding="utf-8").splitlines()
            cases = {
                "missing": source_lines[:-1],
                "duplicate": [*source_lines, source_lines[-1]],
            }
            for name, lines in cases.items():
                with self.subTest(case=name):
                    changed = root / f"{name}.tsv"
                    changed.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    output = root / f"{name}_output"
                    completed = run(
                        "evaluate",
                        "--config",
                        str(CONFIG),
                        "--utilities",
                        str(changed),
                        "--unit-audit-bundle",
                        str(unit_audit),
                        "--output-dir",
                        str(output),
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertFalse(output.exists())

    def test_all_materializers_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="design_audit_create_only_") as temporary:
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
            commands = (
                ("audit-units", "--metadata", str(METADATA)),
                ("plan-comparisons",),
                (
                    "evaluate",
                    "--utilities",
                    str(UTILITIES),
                    "--unit-audit-bundle",
                    str(unit_audit),
                ),
            )
            for index, command in enumerate(commands):
                with self.subTest(command=command[0]):
                    output = root / f"existing_{index}"
                    output.mkdir()
                    marker = output / "OWNER_DATA.txt"
                    marker.write_text("preserve-byte-identically\n", encoding="utf-8")
                    completed = run(
                        command[0],
                        "--config",
                        str(CONFIG),
                        *command[1:],
                        "--output-dir",
                        str(output),
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(marker.read_bytes(), b"preserve-byte-identically\n")
                    self.assertEqual({path.name for path in output.iterdir()}, {marker.name})


if __name__ == "__main__":
    unittest.main(verbosity=2)

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

from jsonschema import Draft202012Validator
import tomli


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY / "analysis/design_audit/quick_demo"
RUNNER = REPOSITORY / "scripts/run_quick_demo.py"
EXPECTED_STDOUT = [
    "Quick demo passed",
    (
        "shared_D: METHOD_A 13.5, METHOD_B 12.5; "
        "METHOD_A > METHOD_B (Holm-adjusted p=0.03125)"
    ),
    (
        "disjoint_D: METHOD_A 12.5, METHOD_B 13.5; "
        "METHOD_B > METHOD_A (Holm-adjusted p=0.03125)"
    ),
    "The descriptive leader changes from METHOD_A to METHOD_B.",
    "units_not_established: WITHHELD (UNIT_INDEPENDENCE_NOT_ESTABLISHED)",
    "Output bundles verified.",
]


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def load_json_strict(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            assert key not in value, f"duplicate JSON key in {path}: {key}"
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    assert isinstance(value, dict)
    return value


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_quick_demo_configs_match_the_published_schema() -> None:
    schema = load_json_strict(
        REPOSITORY / "analysis/design_audit/CONFIG_SCHEMA_V1.json"
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(load_json_strict(FIXTURE / "config.json"))
    validator.validate(
        load_json_strict(FIXTURE / "config_units_not_established.json")
    )


def test_quick_demo_fixture_is_small_hash_bound_and_reverses() -> None:
    positive = load_json(FIXTURE / "config.json")
    negative = load_json(FIXTURE / "config_units_not_established.json")
    utilities_path = FIXTURE / "utilities.tsv"
    observed_hash = hashlib.sha256(utilities_path.read_bytes()).hexdigest()
    assert observed_hash == "4c1bdc205e84f5ec638181601b569336b7cc279533ff430c47f6fa1684522f46"
    assert positive["utility_input_contract"]["expected_input_sha256"] == observed_hash
    assert negative["utility_input_contract"]["expected_input_sha256"] == observed_hash

    positive_copy = json.loads(json.dumps(positive))
    negative_copy = json.loads(json.dumps(negative))
    assert positive_copy.pop("audit_id") == "QUICK_DEMO_REFERENCE_REVERSAL_V1"
    assert negative_copy.pop("audit_id") == "QUICK_DEMO_UNIT_INDEPENDENCE_WITHHELD_V1"
    positive_gate = positive_copy["unit_contract"]["independence_justification"]
    negative_gate = negative_copy["unit_contract"]["independence_justification"]
    assert positive_gate.pop("status") == "ESTABLISHED_BY_EXPERT_ASSERTION"
    assert negative_gate.pop("status") == "NOT_ESTABLISHED"
    assert positive_gate.pop("basis") == "SYNTHETIC_UNITS_ARE_INDEPENDENT_BY_CONSTRUCTION"
    assert negative_gate.pop("basis") == "SYNTHETIC_EXAMPLE_WITHOUT_AN_INDEPENDENCE_BASIS"
    assert positive_copy == negative_copy

    with utilities_path.open(encoding="utf-8", newline="") as stream:
        utilities = list(csv.DictReader(stream, delimiter="\t"))
    with (FIXTURE / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
        metadata = list(csv.DictReader(stream, delimiter="\t"))
    assert len(utilities) == 24
    assert len(metadata) == 12
    assert {row["unit_id"] for row in utilities} == {f"U{i:02d}" for i in range(1, 7)}


def test_quick_demo_runs_to_verified_completion_and_is_create_only() -> None:
    with tempfile.TemporaryDirectory(prefix="test-quick-demo-") as temporary:
        output = Path(temporary) / "first-run"
        command = [
            sys.executable,
            "-I",
            "-s",
            "-B",
            str(RUNNER),
            "--output-root",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        for line in EXPECTED_STDOUT:
            assert line in completed.stdout
        receipt = load_json(output / "QUICK_DEMO_RECEIPT.json")
        assert receipt["status"] == "PASS_QUICK_DEMO"
        assert receipt["shared_D"]["direction"] == "METHOD_A > METHOD_B"
        assert receipt["disjoint_D"]["direction"] == "METHOD_B > METHOD_A"
        assert receipt["units_not_established"] == {
            "decision": "WITHHELD",
            "reason": "UNIT_INDEPENDENCE_NOT_ESTABLISHED",
        }
        assert receipt["separate_bundle_verification"] == "PASS"

        before_repeat = tree_hashes(output)
        repeated = subprocess.run(
            command,
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        assert repeated.returncode == 2
        assert "output path already exists" in repeated.stderr
        assert tree_hashes(output) == before_repeat

        dangling_target = Path(temporary) / "must-not-be-created"
        dangling_link = Path(temporary) / "dangling-output-link"
        os.symlink(dangling_target, dangling_link)
        linked_command = [*command[:-1], str(dangling_link)]
        linked = subprocess.run(
            linked_command,
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        assert linked.returncode == 2
        assert "output path already exists" in linked.stderr
        assert dangling_link.is_symlink()
        assert not dangling_target.exists()


def test_linked_guides_and_ci_run_the_same_protocol_examples() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    guides = "\n".join((REPOSITORY / path).read_text(encoding="utf-8") for path in (
        "docs/organizer.md", "docs/REFERENCE_DESIGN.md",
        "examples/pbmc_reference_protocol/README.md",
    ))
    workflow = (REPOSITORY / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    for command in [
        ".venv/bin/python examples/organizer/make_example.py --output outputs/organizer_inputs",
        ".venv/bin/reference-design design outputs/organizer_inputs/manifest.json --output outputs/organizer_design",
        ".venv/bin/reference-design check-design outputs/organizer_design/design.json outputs/organizer_inputs/assessment.tsv --output outputs/organizer_check",
        ".venv/bin/python examples/anndata_prepare/make_example.py --output outputs/anndata_inputs",
        ".venv/bin/reference-design prepare outputs/anndata_inputs/manifest.json --output outputs/anndata_prepared",
        ".venv/bin/reference-design run outputs/anndata_prepared/plan.json --output outputs/anndata_results",
        ".venv/bin/python examples/pbmc_reference_protocol/run.py --output outputs/pbmc_protocol",
    ]:
        assert guides.count(command) == 1
        assert workflow.count(command) == 1
    assert workflow.count("bash scripts/run_demo.sh") == 1
    assert "ubuntu-latest" in workflow
    assert "macos-15-intel" in workflow
    assert "Python 3.10" in readme

    cli = REPOSITORY / "analysis/design_audit/benchmark_design_audit.py"
    for arguments, expected_text in [
        (["--help"], "acquisition-unit utilities"),
        (["evaluate", "--help"], "unit-by-method utility TSV"),
    ]:
        completed = subprocess.run(
            [sys.executable, "-I", "-s", "-B", str(cli), *arguments],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert expected_text in completed.stdout


def test_supported_package_excludes_archived_model_code() -> None:
    pyproject = tomli.loads(
        (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["requires-python"] == ">=3.10,<3.11"
    discovery = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["src"]
    assert discovery["include"] == [
        "perturb_nuisance_contracts*",
        "perturb_nuisance_focal*",
        "reference_design*",
    ]
    assert discovery["exclude"] == ["perturb_nuisance_model*"]

    source_root = REPOSITORY / discovery["where"][0]
    source_packages = {
        ".".join(path.parent.relative_to(source_root).parts)
        for path in source_root.rglob("__init__.py")
    }
    selected = {
        package
        for package in source_packages
        if any(fnmatch.fnmatchcase(package, pattern) for pattern in discovery["include"])
        and not any(
            fnmatch.fnmatchcase(package, pattern) for pattern in discovery["exclude"]
        )
    }
    assert "perturb_nuisance_model" in source_packages
    assert selected == {"perturb_nuisance_contracts", "perturb_nuisance_focal", "reference_design",
                        "reference_design.reporting_validation"}


def test_formal_qualification_config_and_no_go_record_are_explicit() -> None:
    config = load_json(
        REPOSITORY
        / "configs/qualification/generic_reference_aware_framework_v1_0.json"
    )
    report = load_json(
        REPOSITORY
        / "data/derived/qualification/generic_reference_aware_framework_v1_0/report.json"
    )
    namespace = runpy.run_path(
        str(REPOSITORY / "scripts/qualify_generic_reference_framework_v1.py"),
        run_name="qualification_config_probe",
    )
    assert config == namespace["default_config"]()
    assert len(config["grid"]["root_counts"]) * len(config["grid"]["method_counts"]) * len(
        config["grid"]["profiles"]
    ) == 198
    assert config["simulation"]["repetitions"] == 10_000
    assert report["status"] == "NO_GO_GENERIC_FRAMEWORK_V1_OUTCOME_FREE_VALIDITY"
    assert report["scientific_exit_code"] == 2
    assert report["all_joint_validity_gates_pass"] is False
    failed = [
        scenario
        for scenario in report["scenarios"]
        if scenario["joint_validity_pass"] is not True
    ]
    assert len(failed) == 2


def test_readme_and_ci_run_the_same_one_command_example() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    workflow = (REPOSITORY / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    command = ".venv/bin/reference-design audit demo --output outputs/reference_audit"
    assert readme.count(command) == 1
    assert workflow.count(command) == 1
    assert workflow.count("bash scripts/run_demo.sh") == 1
    assert "ubuntu-latest" in workflow
    assert "macos-15-intel" in workflow
    assert "Python 3.10" in readme

    cli = REPOSITORY / "analysis/design_audit/benchmark_design_audit.py"
    for arguments, expected_text in [
        (["--help"], "acquisition-unit utilities"),
        (["evaluate", "--help"], "unit-by-method utility TSV"),
    ]:
        completed = subprocess.run(
            [sys.executable, "-I", "-s", "-B", str(cli), *arguments],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert expected_text in completed.stdout


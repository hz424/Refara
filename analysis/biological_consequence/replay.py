#!/usr/bin/env python3
"""Reproduce the program analyses and verify their numerical outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(command: list[str]) -> str:
    process = subprocess.run(command, capture_output=True, text=True)
    require(
        process.returncode == 0,
        "Command failed:\n" + " ".join(command) + "\n" + process.stdout + process.stderr,
    )
    return process.stdout


def verify_exact_files(
    directory: Path,
    expected: dict[str, str],
    excluded: set[str] | None = None,
) -> None:
    excluded = excluded or set()
    actual = {path.name for path in directory.iterdir() if path.is_file()} - excluded
    require(actual == set(expected), f"Regenerated membership differs in {directory.name}")
    for name, expected_hash in expected.items():
        require(sha(directory / name) == expected_hash, f"Regenerated file differs: {name}")


def regenerate_interpretation(results: Path, recorded: Path) -> tuple[dict, dict]:
    """Regenerate and exactly compare the post hoc descriptive decomposition."""
    destination = results / "interpretation"
    run([
        sys.executable,
        str(HERE / "summarize_program_interpretation.py"),
        "--program-scores",
        str(results / "program_scores.tsv"),
        "--program-readouts",
        str(results / "program_readouts.npz"),
        "--output-dir",
        str(destination),
    ])

    generated_receipt = json.loads((destination / "INTERPRETATION_RECEIPT.json").read_text())
    require(generated_receipt["status"] == "PASS", "Interpretation decomposition did not pass")
    verify_exact_files(
        destination,
        generated_receipt["outputs_sha256"],
        excluded={"INTERPRETATION_RECEIPT.json"},
    )

    recorded_receipt = json.loads((recorded / "INTERPRETATION_RECEIPT.json").read_text())
    require(recorded_receipt["status"] == "PASS", "Recorded interpretation receipt did not pass")
    expected_names = set(recorded_receipt["outputs_sha256"]) | {"INTERPRETATION_RECEIPT.json"}
    actual_names = {path.name for path in destination.iterdir() if path.is_file()}
    require(actual_names == expected_names, "Interpretation output membership differs")
    for name in sorted(expected_names):
        require(
            sha(destination / name) == sha(recorded / name),
            f"Interpretation output differs: {name}",
        )

    summary = json.loads((destination / "interpretation_summary.json").read_text())
    require(
        summary["status"] == "PASS_POST_HOC_DESCRIPTIVE_SUMMARY",
        "Unexpected interpretation-summary status",
    )
    return generated_receipt, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    require(not args.output_dir.exists(), "Use a new output directory")
    source = args.source_data_root.resolve()
    component = source / "Biological_consequences_v189"
    bindings = json.loads((HERE / "BINDINGS.json").read_text())
    for name, expected_hash in bindings["component_files_sha256"].items():
        require(sha(component / name) == expected_hash, "Source binding differs: " + name)
    for name, expected_hash in bindings["code_files_sha256"].items():
        require(sha(HERE / name) == expected_hash, "Code binding differs: " + name)

    source_validation = json.loads(run([
        sys.executable,
        str(component / "validate.py"),
        "--component-root",
        str(component),
    ]))
    require(
        source_validation["status"] == "PASS_BIOLOGICAL_CONSEQUENCES_V189",
        "Source component did not validate",
    )

    args.output_dir.mkdir(parents=True)
    results = args.output_dir / "results"
    run([
        sys.executable,
        str(HERE / "biological_consequence.py"),
        "--inputs",
        str(component / "inputs/benchmark_inputs.npz"),
        "--axes",
        str(component / "inputs/axes.json"),
        "--programs",
        str(component / "gene_sets/hallmark_mapping.json"),
        "--protocol",
        str(component / "protocol/ANALYSIS_PROTOCOL.md"),
        "--freeze",
        str(component / "protocol/PROTOCOL_FREEZE.json"),
        "--staging-receipt",
        str(component / "INPUT_PROVENANCE.json"),
        "--output-dir",
        str(results),
    ])
    scientific_expected = bindings["scientific_results_sha256"]
    verify_exact_files(
        results,
        scientific_expected,
        excluded={"FIRST_RUN_RECEIPT.json", "analysis_receipt.json"},
    )

    interpretation_receipt, interpretation_summary = regenerate_interpretation(
        results,
        component / "results" / "interpretation",
    )

    supporting = args.output_dir / "supporting"
    run([
        sys.executable,
        str(HERE / "audit_biological_consequence.py"),
        "--inputs",
        str(component / "inputs/benchmark_inputs.npz"),
        "--axes",
        str(component / "inputs/axes.json"),
        "--mapping",
        str(component / "gene_sets/hallmark_mapping.json"),
        "--results",
        str(results),
        "--output-dir",
        str(supporting),
    ])
    supporting_audit = json.loads((supporting / "ALTERNATE_IMPLEMENTATION_AUDIT.json").read_text())
    require(supporting_audit["status"] == "PASS", "Independent arithmetic audit did not pass")
    verify_exact_files(
        supporting,
        bindings["supporting_tables_sha256"],
        excluded={"ALTERNATE_IMPLEMENTATION_AUDIT.json"},
    )

    receipt = {
        "status": "PASS_BIOLOGICAL_CONSEQUENCES_REPLAY",
        "scientific_results_checked": len(scientific_expected),
        "supporting_tables_checked": len(bindings["supporting_tables_sha256"]),
        "post_hoc_interpretation_outputs_checked": len(interpretation_receipt["outputs_sha256"]),
        "post_hoc_interpretation_status": interpretation_summary["status"],
        "source_validation": source_validation,
        "numeric_values_unchanged": True,
        "new_model_fitting": False,
        "new_model_inference": False,
        "run_receipt_note": (
            "New timing and projected-staging-receipt digests distinguish this replay from the "
            "recorded first run. Frozen scientific outputs and the post hoc descriptive "
            "decomposition are exact."
        ),
    }
    (args.output_dir / "REPLAY_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract the compact input level from the matching manuscript SourceData."""
import argparse
import json
from pathlib import Path

from replay import DEPTHS, FAMILIES, INPUT_FIELDS, POLICIES, SEEDS, UNITS
from replay import load_values, read_rows, require, sha256, write_rows

QA_SHA256 = "43cdb270d8f1ff5cf0ab3037553758032f75be394999f7c6779b18cf24df30d7"
PREFIX = "Figure_2_eight_model_v1820/provenance"


def prepare(source_data, output, origin_release):
    source_data, output = Path(source_data).resolve(), Path(output).resolve()
    require(not (output / "SOURCE_MANIFEST.json").exists(), "Refusing to replace existing capsule inputs")
    qa_path = source_data / PREFIX / (QA_SHA256[:16] + "_SOURCE_QA.json")
    require(sha256(qa_path) == QA_SHA256, "Unexpected source QA version")
    qa = json.loads(qa_path.read_text())
    require(qa["status"] == "PASS_FIG2_EIGHT_MODEL_SOURCE_QA", "Source QA did not pass")
    require(qa["families"] == list(FAMILIES) and qa["scored_seeds"] == [17, 29, 43], "Different fitted-model scope")
    require(qa["biological_roots"] == 8 and qa["allocation_labels"] == 1000, "Different evaluation scope")
    sources, groups, records = [], [], []
    reverse_policy = {name: code for code, name in POLICIES.items()}

    def source(pin):
        name = Path(pin["path"]).name
        path = source_data / PREFIX / (pin["sha256"][:16] + "_" + name)
        require(sha256(path) == pin["sha256"], "SourceData hash differs: " + name)
        sources.append({"source_data_path": str(path.relative_to(source_data)),
                        "sha256": pin["sha256"], "bytes": path.stat().st_size})
        return path

    for group in qa["groups"]:
        dataset, regime = group["dataset"], group["regime"]
        require((dataset, regime) in DEPTHS and group["depths"] == list(DEPTHS[dataset, regime]), "Different depth scope")
        pin = next(p for p in group["retained_full_tables"] if Path(p["path"]).name == "ALL_POLICY_ROOT_SEED_MODELS.tsv")
        for row in read_rows(source(pin)):
            if row["panel"] != "primary" or row["seed"] not in SEEDS or row["unit"] not in UNITS:
                continue
            require((row["dataset"], row["regime"]) == (dataset, regime), "Wrong group in source table")
            require(row["allocations"] == "1000" and row["biological_roots"] == "1", "Unexpected score level")
            require(abs(float(row["utility"]) + float(row["mse"])) <= 1e-12, "Utility is not negative MSE")
            compact = {k: row[k] for k in INPUT_FIELDS if k != "policy_code"}
            compact["policy_code"] = reverse_policy[row["policy"]]
            records.append({k: compact[k] for k in INPUT_FIELDS})
        groups.append(dict(dataset=dataset, regime=regime, depths=group["depths"],
            source_model_table_sha256=pin["sha256"],
            evaluation_binding_sha256=group["evaluation_binding"]["sha256"],
            evaluation_completion_sha256=group["completion"]["sha256"],
            full_table_completion_sha256=group["full_table_completion"]["sha256"],
            training_gate_sha256=group["training_gate"]["sha256"]))
    require(len(groups) == 4 and len({g["training_gate_sha256"] for g in groups}) == 1, "Incomplete/common training gate differs")
    load_values(records)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "expected").mkdir(parents=True, exist_ok=True)
    write_rows(output / "data/scored_unit_utilities.tsv", records)

    for source_name, target, fields in (
        ("PRIMARY_ALL_ROOTS_ALL_DEPTHS_MODELS.tsv", "current_mean_utilities.tsv",
         ("dataset", "regime", "depth", "policy_code", "family", "utility")),
        ("PRIMARY_ALL_ROOTS_ALL_DEPTHS_PAIRS.tsv", "current_pair_margins.tsv",
         ("dataset", "regime", "depth", "policy_code", "family_a", "family_b",
          "utility_margin", "shared_utility_margin", "status_exact")),
    ):
        selected = []
        for row in read_rows(source(qa["source_tables"][source_name])):
            if row["unit"] == "equal_eight_root_mean":
                require(row["seed"] == "mean_of_scored_seeds" and row["panel"] == "primary", "Wrong validation level")
                row["policy_code"] = reverse_policy[row["policy"]]
                selected.append({k: row[k] for k in fields})
        write_rows(output / "expected" / target, selected)
    files = {}
    for path in sorted([*(output / "data").glob("*.tsv"), *(output / "expected").glob("*.tsv")]):
        files[str(path.relative_to(output))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = dict(schema="scored-model-comparisons-v1", origin_release=origin_release,
        source_qa={"source_data_path": str(qa_path.relative_to(source_data)), "sha256": QA_SHA256},
        source_tables=sources, groups=groups, files=files, input_rows=len(records),
        selection="primary panel; all three scored seeds and eight units; all registered depths, five designs, eight families",
        input_level="Per-seed/per-unit negative MSE after mean over 1000 balanced equal-depth allocations; no predictions bundled.",
        fitting_provenance="The completed TRAIN-selected fits bound by each evaluation binding and the common training gate are retained unchanged.",
        deterministic_baselines="NC, CM, TW, PCA and RBF repeat the same deterministic score over the three seed labels; this does not create additional independent fits.",
        policy_roles={"S": "model input = prediction reference = observation reference",
                      "M": "prediction reference = observation reference; separate model input",
                      "P": "model input = observation reference; separate prediction reference",
                      "O": "model input = prediction reference; separate observation reference",
                      "D": "three distinct blocks for model input, prediction reference and observation reference"},
        validation_level="Expected files are published primary equal-unit values, used only after recomputing scores, orders and reversal identities from seed/unit inputs.")
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-release", required=True)
    args = parser.parse_args()
    manifest = prepare(args.source_data_root, args.output, args.origin_release)
    print(json.dumps({"input_rows": manifest["input_rows"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()

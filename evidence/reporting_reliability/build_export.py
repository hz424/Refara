"""Export already completed retrospective scores; never fit or score expression.

The source directory is the completed retrospective_gse181897 workspace.
This script is an export utility; replay needs only this capsule and the package.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def table(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def compressed_text(path, value):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
            f.write(value.encode())


def text_table(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent)
    parser.add_argument("--implementation-dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "src/reference_design/reporting_validation")
    args = parser.parse_args()
    src, out = args.source_dir, args.destination
    assert (src / "results/comparison/RESULT.json").is_file(), "Completed result required"
    assert not (out / "source_data").exists(), "Refuse overwriting exported inputs"
    data, expected = out / "source_data", out / "expected"
    data.mkdir(parents=True)
    expected.mkdir()
    freeze = load(src / "design/PROTOCOL_FROZEN.json")
    spec = load(src / "design/statistical_spec.json")
    complete = load(src / "results/score_bank/COMPLETE.json")
    assert complete["status"] == "COMPLETE_SCORE_BANK"
    aliases, source_hashes, array_contracts = {}, {}, {}
    for part in ("C", "P", "A"):
        bank = src / "results/score_bank" / part
        receipt = load(bank / "receipt.json")
        assert sha(bank / "receipt.json") == complete["partitions"][part]["sha256"]
        for name in ("rows.tsv", "losses.npy", "fit_losses.npy", "axes.json"):
            assert sha(bank / name) == receipt["files"][name]["sha256"]
        rows = table(bank / "rows.tsv")
        axes = load(bank / "axes.json")
        mapping = {donor: f"{part}{j:03d}" for j, donor in enumerate(sorted({r['native_donor'] for r in rows}), 1)}
        assert not set(aliases).intersection(mapping), "Partition donor overlap"
        aliases.update(mapping)
        for row in rows:
            row["native_donor"] = mapping[row["native_donor"]]
        compressed_text(data / f"{part}_rows.tsv.gz", text_table(rows))
        losses = np.load(bank / "losses.npy", allow_pickle=False)
        fits = np.load(bank / "fit_losses.npy", allow_pickle=False)
        assert losses.dtype == fits.dtype == np.float64
        for j, family in enumerate(axes["families"]):
            columns = [k for k, item in enumerate(axes["fits"]) if item["family"] == family]
            assert np.array_equal(losses[..., j], fits[..., columns].mean(axis=-1)), "Seed reduction differs"
        path = data / f"{part}_scores.npz"
        np.savez_compressed(path, losses=losses, fit_losses=fits)
        with np.load(path, allow_pickle=False) as check:
            assert np.array_equal(check["losses"], losses) and np.array_equal(check["fit_losses"], fits)
        array_contracts[part] = {"loss_shape": list(losses.shape), "fit_loss_shape": list(fits.shape),
            "dtype": "float64", "lossless_exact_array_roundtrip": True,
            "array_bytes_before_compression": losses.nbytes + fits.nbytes,
            "compressed_bytes": path.stat().st_size, "donors": len(mapping), "tasks": len(rows),
            "fits": axes["fits"], "seed_reduction": axes["seed_reduction"],
            "family_order": axes["families"], "anchor_index": axes["anchor_index"],
            "fold_pairs": axes["fold_pairs"]}
        source_hashes[part] = {name: sha(bank / name) for name in ("rows.tsv", "losses.npy", "fit_losses.npy", "axes.json", "receipt.json")}
    pools = []
    for row in table(freeze["donor_split_path"]):
        if row["donor"] in aliases:
            pools.append({"donor": aliases[row["donor"]], **{k: row[k] for k in ("pool_control", "pool_beta", "pool_gamma", "pool_tnf")}})
    compressed_text(data / "donor_pools.tsv.gz", text_table(pools))
    for name in ("summary.tsv", "work_accounting.tsv"):
        compressed_text(expected / (name + ".gz"), (src / "results/comparison" / name).read_text())
    pool_expected = table(src / "results/comparison/pool_sensitivity.tsv")
    for row in pool_expected:
        row["excluded_donors"] = ",".join(aliases[d] for d in row["excluded_donors"].split(",") if d)
    compressed_text(expected / "pool_sensitivity.tsv.gz", text_table(pool_expected))
    intervals = load(src / "results/comparison/assessment_intervals.json")
    for panel in intervals.values():
        for row in panel["diagnostics"]:
            row["donors"] = [aliases[d] for d in row["donors"]]
    write_json(expected / "assessment_intervals.json", intervals)
    audited = load(src / "qa/scientific_conclusion.json")
    write_json(expected / "coverage.json", {panel: {k: x[k] for k in ("primary_M", "primary_status", "common_achievable_M", "all_common_M")}
                                           for panel, x in audited["panels"].items()})
    protocol = {"schema": "refara.reporting_reliability.retrospective.v1",
        "mode": "retrospective_used_data_only", "scientific_confirmation": False,
        "prior_exposure": "All datasets and historical assessment outcomes were used or inspected before this comparison; the original P16/A46 assignment is retained. This is not untouched validation.",
        "scope": spec["scope"], "rosters": spec["rosters"],
        "score_contract": spec["score_contract"], "five_rules": spec["five_rules"],
        "coverage": spec["coverage"], "primary_assessment": spec["primary_assessment"],
        "secondary_assessment": spec["secondary_assessment"], "pool_sensitivity": spec["pool_sensitivity"],
        "classification": spec["classification"], "first_two_crossfit_companion": spec["first_two_crossfit_companion"],
        "local_freeze_utc": freeze["frozen_utc"], "public_preregistration": False,
        "original_protocol_sha256": sha(src / "design/PROTOCOL_FROZEN.json"),
        "original_result_sha256": audited["result_sha256"],
        "source_score_hashes": source_hashes, "array_contracts": array_contracts,
        "source_urls": ["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181897", "https://www.parsebiosciences.com/datasets/"],
        "publication_bootstrap_resamples": 9999, "publication_bootstrap_seed": 20260922,
        "assessment_bootstrap_resamples": 19999, "assessment_bootstrap_seed": 20260923,
        "numerical_expected_tolerance": {"absolute": 1e-12, "relative": 1e-12,
            "reason": "SciPy t-quantiles may differ in final floating-point bits. Decisions use the original strict zero tests, never this comparison tolerance."},
        "compression_scope": "All float64 family and per-fit scored losses, row ordering, allocation ordering and pool membership needed for the reporting comparison are preserved exactly. Raw expression, per-gene predictions, checkpoints, cell identifiers, and repeated per-coverage JSON detail are excluded. This replays saved scores, not model inference.",
        "native_unit_caveat": "Rows are donors under a conditional working-independence model within existing experimental pools. Donor identity does not prove independent culture replication; intervals and finite-panel bounds are assumption-conditional diagnostics, not validated coverage guarantees or new-pool inference."}
    def portable(value):
        if isinstance(value, dict):
            return all(portable(v) for v in value.values())
        if isinstance(value, list):
            return all(portable(v) for v in value)
        return not (isinstance(value, str) and value.startswith(("/", "~/")))
    assert portable(protocol), "Absolute path in portable protocol"
    write_json(out / "protocol.json", protocol)
    total_raw = sum(r["array_bytes_before_compression"] for r in array_contracts.values())
    total_zip = sum(r["compressed_bytes"] for r in array_contracts.values())
    write_json(out / "EXPORT_RECEIPT.json", {"status": "LOSSLESS_SCORE_EXPORT", "raw_array_bytes": total_raw,
        "compressed_array_bytes": total_zip, "compression_ratio": total_zip / total_raw,
        "original_result_bytes": (src / "results/comparison/RESULT.json").stat().st_size,
        "raw_expression_read": False, "model_inference_performed": False,
        "original_independent_audit_sha256": sha(src / "qa/completed_result_independent_audit.json"),
        "original_audit_status": load(src / "qa/completed_result_independent_audit.json")["status"]})
    model_contract = Path(__file__).parent.parent / "gse181897_reference_design/model_contract.json"
    if model_contract.is_file():
        shutil.copy2(model_contract, out / "model_contract.json")
    assert (args.implementation_dir / "replay.py").is_file(), "Packaged replay implementation required"
    write_json(out / "code_contract.json", {p.name: sha(p) for p in sorted(args.implementation_dir.glob("*.py"))})
    write_json(out / "MANIFEST.json", {"schema": "refara.reporting_reliability.manifest.v1",
        "files": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.name != "MANIFEST.json"}})
    print(json.dumps({"raw_bytes": total_raw, "compressed_bytes": total_zip}))


if __name__ == "__main__":
    main()

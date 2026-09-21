from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import reference_design


def _write_table(path, columns, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)


def _read_table(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _vector(path, values, order=("g1", "g2", "g3")):
    _write_table(path, ["gene", "value"], [(gene, values[gene]) for gene in order])


def _run(*arguments, succeeds=True):
    env = os.environ.copy()
    package_parent = str(Path(reference_design.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "reference_design", *map(str, arguments)],
        text=True, capture_output=True, env=env, check=False,
    )
    if succeeds:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result


@pytest.fixture
def case(tmp_path):
    cells = tmp_path / "controls.tsv"
    _write_table(
        cells, ["cell_id", "stratum", "g3", "g1", "g2"],
        [(f"cell_{index}", "capture", -(index - 4), index - 4, 2 * (index - 4)) for index in range(9)],
    )
    allocation = tmp_path / "allocation"
    _run("allocate", cells, "--depth", "1", "--depth", "3", "--allocations", "2", "--seed", "12", "--output", allocation)
    _vector(tmp_path / "treated.tsv", {"g1": 0.0, "g2": 0.0, "g3": 0.0})
    _vector(tmp_path / "scales.tsv", {"g1": 1.0, "g2": 2.0, "g3": 3.0})
    _vector(tmp_path / "effect.tsv", {"g1": 0.1, "g2": -0.2, "g3": 0.05})
    _vector(tmp_path / "bad_effect.tsv", {"g1": 9.0, "g2": 9.0, "g3": 9.0})
    _vector(tmp_path / "state.tsv", {"g1": 0.0, "g2": 0.0, "g3": 0.0})
    config = {
        "scales": "scales.tsv",
        "tasks": [{
            "id": "task", "unit": "donor",
            "treated": "treated.tsv", "references": "allocation/references.tsv",
            "controls": "controls.tsv", "membership": "allocation/membership.tsv",
            "models": [
                {"name": "effect", "kind": "effect", "prediction": "effect.tsv"},
                {"name": "bad effect", "kind": "effect", "prediction": "bad_effect.tsv"},
                {"name": "state", "kind": "state", "prediction": "state.tsv"},
            ],
        }],
    }
    return tmp_path, config


def _score(case, directory="result", succeeds=True):
    root, config = case
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = root / directory
    result = _run("score", path, "--output", output, succeeds=succeeds)
    return output, result


def _score_values(path):
    rows = _read_table(path / "scores.tsv")
    return {
        (row["task"], row["allocation"], row["depth"], row["pattern"], row["model"]): float(row["MSE"])
        for row in rows
    }


def test_cells_to_complete_pair_tables_and_figure(case):
    root, _ = case
    allocation = json.loads((root / "allocation" / "allocation.json").read_text())
    assert allocation["seed"] == 12
    assert allocation["allocations"] == 2
    assert allocation["depths"] == [1, 3]
    assert allocation["strata_count"] == 1
    assert allocation["numpy_version"] == np.__version__
    assert allocation["controls_sha256"] == hashlib.sha256((root / "controls.tsv").read_bytes()).hexdigest()
    output, _ = _score(case)
    assert {"scores.tsv", "pairs.tsv", "checks.json", "sensitivity.pdf"} <= {path.name for path in output.iterdir()}
    assert len(_read_table(output / "scores.tsv")) == 2 * 2 * 5 * 3
    pairs = _read_table(output / "pairs.tsv")
    assert len(pairs) == 2 * 2 * 3
    assert {row["unit"] for row in pairs} == {"donor"}
    assert {row["theory"] for row in pairs} == {"reference_penalty", "zero_displacement"}
    assert all(abs(float(row["identity_residual"])) < 1e-11 for row in pairs)
    assert not any("p_value" in key or "confidence" in key for key in pairs[0])
    checks = json.loads((output / "checks.json").read_text())
    expected_inputs = [root / name for name in (
        "config.json", "controls.tsv", "allocation/references.tsv", "allocation/membership.tsv",
        "treated.tsv", "scales.tsv", "effect.tsv", "bad_effect.tsv", "state.tsv",
    )]
    assert checks["inputs"] == sorted(
        [dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
         for path in expected_inputs], key=lambda item: item["path"],
    )
    assert checks["identity_checks"][0]["reference_check"] == "verified_membership"
    assert all(binding["level"] == "declaration" and not binding["verified_input_use"]
               for binding in checks["identity_checks"][0]["input_bindings"])
    mixed = [row for row in pairs if row["model_a"] == "effect" and row["model_b"] == "state" and row["depth"] == "3"]
    assert len(mixed) == 2
    assert all(float(row["d_S"]) < 0 < float(row["d_D"]) for row in mixed)
    from pypdf import PdfReader
    pdf = PdfReader(output / "sensitivity.pdf")
    assert len(pdf.pages) >= 1
    text = " ".join(" ".join(page.extract_text().split()) for page in pdf.pages)
    assert "effect vs state" in text
    assert "positive favours first model" in text
    assert len(list(output.glob("*.pdf"))) == 1


def test_supplied_means_are_distinguished_from_verified_cell_membership(case):
    _, config = case
    verified, _ = _score(case, "verified")
    task = config["tasks"][0]
    for key in ("controls", "membership", "unit"):
        task.pop(key)
    supplied, _ = _score(case, "supplied")
    assert _score_values(verified) == _score_values(supplied)
    check = json.loads((supplied / "checks.json").read_text())["identity_checks"][0]
    assert check["reference_check"] == "supplied_means"
    assert check["unit"] == ""


def test_gene_order_is_aligned_across_every_table(case):
    root, _ = case
    original, _ = _score(case, "original")
    for name in ("treated.tsv", "scales.tsv", "effect.tsv", "bad_effect.tsv", "state.tsv"):
        path = root / name
        values = {row["gene"]: row["value"] for row in _read_table(path)}
        _vector(path, values, order=("g3", "g2", "g1"))
    references = root / "allocation" / "references.tsv"
    rows = _read_table(references)
    columns = [key for key in rows[0] if key not in ("g1", "g2", "g3")] + ["g2", "g3", "g1"]
    _write_table(references, columns, [[row[key] for key in columns] for row in rows])
    permuted, _ = _score(case, "permuted")
    a, b = _score_values(original), _score_values(permuted)
    assert a.keys() == b.keys()
    np.testing.assert_allclose(list(a.values()), [b[key] for key in a], atol=1e-12)


@pytest.mark.parametrize("problem", ["missing_gene", "duplicate_gene", "duplicate_header"])
def test_malformed_gene_axes_fail_before_writing_outputs(case, problem):
    root, _ = case
    if problem == "missing_gene":
        _write_table(root / "effect.tsv", ["gene", "value"], [("g1", 0.1), ("g2", -0.2)])
    elif problem == "duplicate_gene":
        _write_table(root / "effect.tsv", ["gene", "value"], [("g1", 0.1), ("g1", -0.2), ("g3", 0.05)])
    else:
        path = root / "allocation" / "references.tsv"
        lines = path.read_text().splitlines()
        lines[0] = lines[0].replace("g2", "g1")
        path.write_text("\n".join(lines) + "\n")
    output, _ = _score(case, succeeds=False)
    assert not output.exists()


@pytest.mark.parametrize("level", ["root", "task", "model"])
def test_unknown_configuration_fields_are_rejected(case, level):
    _, config = case
    if level == "root":
        target = config
    elif level == "task":
        target = config["tasks"][0]
    else:
        target = config["tasks"][0]["models"][0]
    target["scorse"] = "misspelled setting"
    output, _ = _score(case, succeeds=False)
    assert not output.exists()


def _conditioned_tables(case):
    root, config = case
    references = _read_table(root / "allocation" / "references.tsv")
    columns = ["allocation", "depth", "block", "g1", "g2", "g3"]
    states, effects, baselines = [], [], []
    for row in reversed(references):
        key = [row["allocation"], row["depth"], row["block"]]
        baseline = np.array([float(row[gene]) for gene in ("g1", "g2", "g3")]) + np.array([0.4, -0.2, 0.7])
        state = baseline * 1.7 + np.array([0.3, -0.5, 0.2])
        states.append(key + state.tolist())
        effects.append(key + (state - baseline).tolist())
        baselines.append(key + baseline.tolist())
    _write_table(root / "conditioned_state.tsv", columns, states)
    _write_table(root / "conditioned_effect.tsv", columns, effects)
    _write_table(root / "baseline.tsv", columns, baselines)
    config["tasks"][0]["models"][2].update(prediction="conditioned_state.tsv", conditioning="block")


def test_conditioned_state_keys_and_explicit_baseline_conversion(case):
    root, config = case
    _conditioned_tables(case)
    state_result, _ = _score(case, "state_result")
    model = config["tasks"][0]["models"][2]
    model.update(prediction="conditioned_effect.tsv", representation="effect", baseline="baseline.tsv")
    converted_result, _ = _score(case, "converted_result")
    input_hashes = {item["path"]: item["sha256"]
                    for item in json.loads((converted_result / "checks.json").read_text())["inputs"]}
    for name in ("conditioned_effect.tsv", "baseline.tsv"):
        path = root / name
        assert input_hashes[str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
    a, b = _score_values(state_result), _score_values(converted_result)
    assert a.keys() == b.keys()
    np.testing.assert_allclose(list(a.values()), [b[key] for key in a], atol=1e-12)
    state_pairs = _read_table(state_result / "pairs.tsv")
    effect_pairs = _read_table(converted_result / "pairs.tsv")
    for state_row, effect_row in zip(state_pairs, effect_pairs):
        assert state_row["kind_a"] == effect_row["kind_a"]
        assert state_row["kind_b"] == effect_row["kind_b"]
        assert float(state_row["d_D"]) == pytest.approx(float(effect_row["d_D"]), abs=1e-12)


@pytest.mark.parametrize("problem", ["conditioned_effect", "converted_native_effect", "missing_state_baseline"])
def test_output_semantics_are_not_silently_reinterpreted(case, problem):
    _, config = case
    model = config["tasks"][0]["models"][0]
    if problem == "conditioned_effect":
        model["conditioning"] = "block"
    elif problem == "converted_native_effect":
        model.update(representation="effect", baseline="state.tsv")
    else:
        model.update(kind="state", representation="effect")
    output, _ = _score(case, succeeds=False)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown_block"])
def test_conditioning_table_requires_exact_reference_keys(case, problem):
    root, _ = case
    _conditioned_tables(case)
    path = root / "conditioned_state.tsv"
    rows = _read_table(path)
    if problem == "missing":
        rows = rows[:-1]
    elif problem == "duplicate":
        rows.append(rows[0].copy())
    else:
        rows[0]["block"] = "B4"
    columns = list(rows[0])
    _write_table(path, columns, [[row[key] for key in columns] for row in rows])
    output, _ = _score(case, succeeds=False)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["shared_cell", "missing_membership", "wrong_mean", "wrong_count"])
def test_membership_and_reference_means_are_verified(case, problem):
    root, _ = case
    if problem in ("shared_cell", "missing_membership"):
        path = root / "allocation" / "membership.tsv"
        rows = _read_table(path)
        if problem == "missing_membership":
            rows = rows[:-1]
        else:
            first = rows[0]
            other = next(row for row in rows if row["allocation"] == first["allocation"] and row["depth"] == first["depth"] and row["block"] != first["block"])
            other["cell_id"] = first["cell_id"]
    else:
        path = root / "allocation" / "references.tsv"
        rows = _read_table(path)
        key = "g1" if problem == "wrong_mean" else "cells_per_block"
        rows[0][key] = str(float(rows[0][key]) + 1)
    columns = list(rows[0])
    _write_table(path, columns, [[row[key] for key in columns] for row in rows])
    output, _ = _score(case, succeeds=False)
    assert not output.exists()


def test_late_invalid_task_does_not_leave_partial_output(case):
    _, config = case
    bad = json.loads(json.dumps(config["tasks"][0]))
    bad["id"] = "later_task"
    bad["treated"] = "nonexistent.tsv"
    config["tasks"].append(bad)
    output, _ = _score(case, succeeds=False)
    assert not output.exists()


def test_existing_outputs_are_preserved(case):
    root, _ = case
    output = root / "result"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("existing result\n")
    _, _ = _score(case, succeeds=False)
    assert marker.read_text() == "existing result\n"
    assert [path.name for path in output.iterdir()] == ["keep.txt"]


def test_native_effect_block_conditioning_scores_without_prediction_centring(case):
    from itertools import product
    root, config = case
    rows = _read_table(root/'allocation/references.tsv')
    columns = ['allocation', 'depth', 'block', 'g1', 'g2', 'g3']
    _write_table(root/'block_effect.tsv', columns, [[row[k] for k in columns] for row in rows])
    config['tasks'][0]['models'][0].update(conditioning='block', prediction='block_effect.tsv')
    output, _ = _score(case)
    scores = _read_table(output/'scores.tsv')
    for allocation, depth in {(r['allocation'], r['depth']) for r in rows}:
        controls = {r['block']: np.array([float(r[g]) for g in ['g1','g2','g3']])
                    for r in rows if r['allocation']==allocation and r['depth']==depth}
        selected = [r for r in scores if r['model']=='effect' and r['allocation']==allocation and r['depth']==depth]
        for row in selected:
            expected = []
            for obs, pred, inp in product(range(3), repeat=3):
                pattern = 'S' if obs==pred==inp else 'M' if obs==pred else 'P' if obs==inp else 'O' if pred==inp else 'D'
                if pattern == row['pattern']:
                    expected.append(np.mean(((controls[f'B{inp+1}']+controls[f'B{obs+1}'])/[1,2,3])**2))
            assert float(row['MSE']) == pytest.approx(np.mean(expected))
    pairs = _read_table(output/'pairs.tsv')
    affected = [r for r in pairs if 'effect' in (r['model_a'], r['model_b'])]
    assert affected and all(r['identity_applicable']=='False' for r in affected)
    assert all(r['identity_residual']=='' for r in affected)
    checks = json.loads((output/'checks.json').read_text())['identity_checks'][0]
    assert checks['identities_unavailable'] == len(affected)
    assert all(b['level']=='declaration' for b in checks['input_bindings'])


@pytest.mark.parametrize("filename", ["treated.tsv", "scales.tsv"])
def test_score_receipt_tracks_changed_target_and_scales(case, filename):
    root, _ = case
    before, _ = _score(case, "before")
    path = root / filename
    rows = _read_table(path)
    rows[0]["value"] = str(float(rows[0]["value"]) + 1)
    _write_table(path, ["gene", "value"], [[row["gene"], row["value"]] for row in rows])
    after, _ = _score(case, "after")
    hashes = [{row["path"]: row["sha256"] for row in
               json.loads((output / "checks.json").read_text())["inputs"]}
              for output in (before, after)]
    assert hashes[0].keys() == hashes[1].keys()
    assert {name for name in hashes[0] if hashes[0][name] != hashes[1][name]} == {str(path.resolve())}
    assert hashes[1][str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert _score_values(before) != _score_values(after)


def test_score_receipt_includes_generation_record_dependencies(case):
    root, config = case
    from reference_design.input_binding import SCHEMA

    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(b"synthetic checkpoint identity")
    code = root / "generate.py"
    code.write_text("# synthetic generation identity, not executed\n")

    def pin(name):
        return dict(path=name, sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())

    record = dict(
        schema=SCHEMA,
        model=dict(name="state", kind="state", conditioning="fixed", representation="native"),
        artifacts=dict(prediction=pin("state.tsv"), controls=pin("controls.tsv"),
                       membership=pin("allocation/membership.tsv"),
                       references=pin("allocation/references.tsv"), baseline=None),
        input_keys=[], checkpoint=pin("checkpoint.bin"), code=[pin("generate.py")],
    )
    record_path = root / "generation.json"
    record_path.write_text(json.dumps(record))
    config["tasks"][0]["models"][2]["generation_record"] = record_path.name
    output, _ = _score(case)
    checks = json.loads((output / "checks.json").read_text())
    hashes = {item["path"]: item["sha256"] for item in checks["inputs"]}
    for path in (checkpoint, code, record_path, root / "treated.tsv", root / "scales.tsv"):
        assert hashes[str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
    binding = next(item for item in checks["identity_checks"][0]["input_bindings"]
                   if item["model"] == "state")
    assert binding["level"] == "generation_record_binding"
    assert binding["verified_input_use"] is False
    assert binding["generation_model"] == record["model"]


def test_changed_input_during_scoring_is_not_published(case, monkeypatch):
    from argparse import Namespace
    from reference_design import cli

    root, config = case
    path = root / "config.json"
    path.write_text(json.dumps(config))
    output = root / "result"

    def changed_input(_pairs):
        (root / "treated.tsv").write_text("changed after scoring\n")
        return b"unpublished plot"

    monkeypatch.setattr(cli, "_figure", changed_input)
    with pytest.raises(ValueError, match="Scoring input changed during the run"):
        cli._score(Namespace(config=path, output=output))
    assert not output.exists()

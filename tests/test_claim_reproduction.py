"""Check mechanism scope and failures in the numerical reproduction entry point."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location('claim_' + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_identity_and_counterexample():
    result = load('evidence/reference_roles/replay.py').recompute()
    assert [r['reversal'] for r in result['cases']] == [True, False, False]
    assert result['cases'][0]['d_S'] == pytest.approx(-2 / 3)
    assert result['cases'][0]['d_D'] == pytest.approx(4 / 3)
    assert result['equivalent_representation_max_abs'] < 1e-12
    counter = result['input_dependent_effect_counterexample']
    assert counter['d_D'] - counter['d_S'] == pytest.approx(4)
    assert counter['V'] == pytest.approx(2)
    assert counter['identity_applicable'] is False


def test_claim_runner_writes_real_result_and_preserves_existing_output(tmp_path):
    runner = load('evidence/reproduce.py')
    output = tmp_path / 'results'
    report = runner.reproduce(output, ['reference_roles'])
    assert report['status'] == 'PASS_IMPORTANT_CONCLUSIONS'
    before = (output / 'RESULTS.json').read_bytes()
    assert json.loads(before)['claims']['reference_roles']['result']['cases'][0]['reversal']
    with pytest.raises(ValueError, match='new output'):
        runner.reproduce(output, ['reference_roles'])
    assert (output / 'RESULTS.json').read_bytes() == before


def test_failure_does_not_report_all_claims_passed(tmp_path, monkeypatch):
    runner = load('evidence/reproduce.py')
    monkeypatch.setitem(runner.CLAIMS, 'broken', ('Missing component', 'does-not-exist.py', 'REPLAY.json'))
    output = tmp_path / 'failure'
    with pytest.raises(RuntimeError, match='broken failed'):
        runner.reproduce(output, ['reference_roles', 'broken'])
    report = json.loads((output / 'RESULTS.json').read_text())
    assert report['status'] == 'FAIL_IMPORTANT_CONCLUSIONS'
    assert set(report['claims']) == {'reference_roles'}

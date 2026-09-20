"""Invalid audit input should produce a command-line error without publishing results."""

import json

import pytest

from reference_design.cli import main


@pytest.mark.parametrize("problem", ["missing_file", "invalid_json", "invalid_contract", "existing_output"])
def test_audit_plan_input_errors_are_reported_without_tracebacks(tmp_path, capsys, problem):
    contract = dict(task="reference_effect", output_kind="state", input_conditioned=False,
                    storage_encoding="state", prediction_reference_id="B1", observation_reference_id="B2")
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps(contract))
    after.write_text(json.dumps(contract))
    output = tmp_path / "audit"
    if problem == "missing_file":
        before.unlink()
    elif problem == "invalid_json":
        before.write_text("{invalid json")
    elif problem == "invalid_contract":
        before.write_text(json.dumps(dict(contract, model_inpt_id="B3")))
    else:
        output.mkdir()
        (output / "keep.txt").write_text("existing output\n")

    with pytest.raises(SystemExit) as error:
        main(["audit", "plan", str(before), str(after), "--output", str(output)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "reference-design audit: error:" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out
    if problem == "existing_output":
        assert (output / "keep.txt").read_text() == "existing output\n"
        assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]
    else:
        assert not output.exists()

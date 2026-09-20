"""Independent rank examples and damaged-input checks for the current capsule."""
from pathlib import Path
import importlib.util
import math
import shutil
import pytest
from scipy.stats import kendalltau

CAPSULE=Path(__file__).resolve().parents[1]/'evidence/current_submission/ed9'
SPEC=importlib.util.spec_from_file_location('current_ed9',CAPSULE/'replay.py')
replay=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

@pytest.mark.parametrize('left,right',[
    ([0.,1.,2.],[2.,1.,0.]),
    ([0.,0.,2.],[0.,2.,2.]),
    ([1.,1.,1.],[2.,1.,0.]),
    ([1.,2.,2.,4.],[4.,2.,2.,1.]),
])
def test_kendall_with_ties_matches_independent_scipy(left,right):
    actual=replay.tau_b(left,right)
    expected=kendalltau(left,right).statistic
    if math.isnan(expected):assert actual is None
    else:assert actual==pytest.approx(expected,abs=1e-15)

def test_equal_scores_have_midrank_and_nonfinite_scores_fail():
    assert replay.ranks([3.,1.,1.,2.])==[4.,1.5,1.5,3.]
    with pytest.raises(ValueError,match='Nonfinite'):replay.ranks([1.,float('nan')])

def test_changed_metric_input_is_rejected_before_ranking(tmp_path):
    target=tmp_path/'ed9'
    shutil.copytree(CAPSULE,target)
    path=target/'primary_metric_scores.tsv'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Input hash differs'):replay.recompute(target)

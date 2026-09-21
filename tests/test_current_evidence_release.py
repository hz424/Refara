"""Bounded numerical replay and rejection of altered scientific inputs."""
import importlib.util,json,shutil
from pathlib import Path
import pytest
REPO=Path(__file__).resolve().parents[1]


def load(relative):
    spec=importlib.util.spec_from_file_location('current_evidence_'+Path(relative).stem,REPO/relative)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_current_figure3_seed_and_root_reaggregation():
    plot,pairs,result=load('evidence/figure3/replay.py').recompute()
    assert len(plot)==576 and len(pairs)==96
    assert result['max_abs_reconstruction_difference']<2e-12
    assert result['all_pair_depth_comparisons']=={'D_to_O':48,'D_to_P':48}
    assert not result['historical_representation_oracles']['reexecuted']


def test_figure3_replay_rejects_modified_source_before_computation(tmp_path):
    capsule=REPO/'evidence/figure3';shutil.copytree(capsule,tmp_path/'capsule')
    target=tmp_path/'capsule/expected/FIGURE3_PRIMARY_ROOT_VALUES.tsv'
    target.write_text(target.read_text().replace('0.08805480339153374','0.18805480339153374',1))
    with pytest.raises(ValueError,match='Changed source'):load('evidence/figure3/replay.py').recompute(tmp_path/'capsule')


def test_current_table1_selection_uses_correct_statistic_and_fixed_output_controls():
    evidence=REPO/'evidence';selection=json.loads((evidence/'TABLE1_CURRENT_SELECTION.json').read_text())
    result=load('evidence/replay_table1.py').check_selection(evidence/'table1/results/role_contrasts.tsv',selection)
    assert len(result)==16
    assert sum(r['model']!='selected_CPA' and float(r['mean_abs_task_mean_change'])==0 for r in result)==10

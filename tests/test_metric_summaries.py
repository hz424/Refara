"""Missing controls and incomplete root sets must not produce usable scores."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/'analysis/metric_summaries/validate.py'
SPEC = importlib.util.spec_from_file_location('metric_summaries_validation', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row():
    return dict(protocol_id=MODULE.PEARSON, tier='example', resource_id='resource',
                depth='16', pattern_id='ALL_SHARED', method_id='example_model',
                direction='HIGHER_IS_BETTER', allocation_count='1000',
                status='PASS_RESOURCE_SUMMARY', complete_family_status='INCOMPLETE',
                failure_reason='', root_count='8',
                **{name:'0.25' for name in MODULE.NUMERIC})


def test_defined_model_can_belong_to_incomplete_family():
    assert MODULE.check_row(row())
    nc = row()
    nc.update(method_id=MODULE.NC, status='UNDEFINED_MODEL_SUMMARY',
              failure_reason='UNDEFINED_ZERO_EFFECT_PEARSON')
    nc.update({name:'' for name in MODULE.NUMERIC})
    assert MODULE.check_row(nc) is False


def test_finite_no_change_correlation_is_rejected():
    nc = row()
    nc['method_id'] = MODULE.NC
    with pytest.raises(ValueError, match='Zero-effect Pearson'):
        MODULE.check_row(nc)


def test_incomplete_root_set_cannot_be_renormalized():
    resource = row()
    roots = [dict(resource, root_id=str(i)) for i in range(7)]
    with pytest.raises(ValueError, match='all eight distinct roots'):
        MODULE.check_root_means([resource], roots)


def test_undefined_root_cannot_yield_finite_resource_mean():
    resource = row()
    roots = [dict(resource, root_id=str(i)) for i in range(8)]
    roots[-1].update({name:'' for name in MODULE.NUMERIC})
    with pytest.raises(ValueError, match='Incomplete roots'):
        MODULE.check_root_means([resource], roots)


def test_resource_mean_uses_all_roots():
    resource = row()
    roots = [dict(resource, root_id=str(i)) for i in range(8)]
    roots[-1]['raw_mean'] = '0.75'
    with pytest.raises(ValueError, match='equal eight-root mean'):
        MODULE.check_root_means([resource], roots)


def test_original_numeric_text_is_preserved_exactly():
    original = row()
    original.update(complete_family_status='COMPLETE', complete_family_failure_reason='')
    restored = dict(original, raw_mean='0.250')
    with pytest.raises(ValueError, match='Previously populated summary changed'):
        MODULE.compare_tables([original], [restored], 'resource')


def test_restored_score_does_not_qualify_incomplete_family():
    original = row()
    original.update(status='COMPACT_COMPLETE_FAMILY_NONPASS',
                    failure_reason='COMPACT_COMPLETE_FAMILY_NONPASS')
    original.update({name:'' for name in MODULE.NUMERIC})
    restored = row()
    restored.update(complete_family_status='COMPLETE',
                    complete_family_failure_reason='COMPACT_COMPLETE_FAMILY_NONPASS')
    with pytest.raises(ValueError, match='Original family eligibility changed'):
        MODULE.compare_tables([original], [restored], 'resource')

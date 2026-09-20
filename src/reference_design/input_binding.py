"""Check generation-record consistency without attesting historical generation.

A JSON record cannot promote itself to independent verification. An optional
trusted Python callback must actually return recomputed numeric predictions.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

SCHEMA = 'reference_design.generation_record.v1'
_CONTEXT = {'name', 'kind', 'conditioning', 'representation'}
_ROLES = {'prediction', 'controls', 'membership', 'references', 'baseline'}


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'Duplicate generation-record field: {key}')
        result[key] = value
    return result


def _fields(value, expected, where):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'{where}: expected exactly {sorted(expected)}')


def _keys(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError('input_keys must be a list of allocation/depth/block keys')
    result = []
    for key in values:
        if (not isinstance(key, (list, tuple)) or len(key) != 3
                or not isinstance(key[0], str) or not key[0]
                or type(key[1]) is not int or key[1] < 1
                or key[2] not in ('B1', 'B2', 'B3')):
            raise ValueError('Invalid allocation/depth/block input key')
        result.append(tuple(key))
    if len(result) != len(set(result)):
        raise ValueError('Duplicate input key')
    return sorted(result)


def _numeric(value):
    if isinstance(value, dict):
        if not value:
            raise ValueError('Recomputed prediction mapping must be nonempty')
        return {key: _numeric(array) for key, array in value.items()}
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not array.size or not np.isfinite(array).all():
        raise ValueError('Recompute callback must return finite numeric predictions, not a status')
    return array


def _equal_numeric(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return False
        return all(_equal_numeric(expected[key], actual[key]) for key in expected)
    return not isinstance(actual, dict) and expected.shape == actual.shape and np.array_equal(expected, actual)


def verify_generation_record(
    record_path, *, prediction_path, model_name, kind, conditioning,
    representation='native', controls_path=None, membership_path=None,
    references_path=None, baseline_path=None, expected_input_keys=None,
    recompute: Callable | None = None, expected_predictions=None,
):
    """Validate a record against files actually selected by a scoring caller.

    ``record_path=None`` retains declaration-only operation. Otherwise all named
    artifact hashes, exact model context and complete input keys must agree.
    Block-conditioned records require raw controls, membership and references;
    the caller must also perform its normal membership/mean validation.

    A trusted application may pass ``recompute(record)`` and the numeric values
    it will score. Only an executed callback with exact matching arrays earns
    ``independently_recomputed``. Callbacks are never loaded from JSON. This
    verifies agreement with that callback, not the historical act of generation
    or training-data exclusion. There is deliberately no caller-provided PASS
    field, tolerance or independent-verification label in the record schema.
    """
    if record_path is None:
        if recompute is not None or expected_predictions is not None:
            raise ValueError('Recomputation requires a bound generation record')
        return dict(level='declaration', verified_input_use=False, artifacts={})
    path = Path(record_path).resolve(strict=True)
    record = json.loads(path.read_text(), object_pairs_hook=_pairs,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Invalid JSON number {x}')))
    _fields(record, {'schema', 'model', 'artifacts', 'input_keys', 'checkpoint', 'code'}, 'Generation record')
    if record['schema'] != SCHEMA:
        raise ValueError('Unsupported generation-record schema')
    _fields(record['model'], _CONTEXT, 'Generation model')
    context = dict(name=model_name, kind=kind, conditioning=conditioning, representation=representation)
    if record['model'] != context:
        raise ValueError('Generation-record model context differs from scored model')
    _fields(record['artifacts'], _ROLES, 'Generation artifacts')
    actual = dict(prediction=prediction_path, controls=controls_path, membership=membership_path,
                  references=references_path, baseline=baseline_path)
    if prediction_path is None:
        raise ValueError('Scored prediction path is required')
    if conditioning == 'block' and any(actual[key] is None for key in ('controls', 'membership', 'references')):
        raise ValueError('Block generation-record binding requires controls, membership and references')
    expected_keys = _keys(expected_input_keys or [])
    if conditioning == 'block' and not expected_keys:
        raise ValueError('Block binding requires complete input keys')
    if conditioning == 'fixed' and expected_keys:
        raise ValueError('Fixed prediction must not declare block input keys')
    if _keys(record['input_keys']) != expected_keys:
        raise ValueError('Generation-record input keys differ from scored inputs')
    artifacts = {}

    def bind(name, spec, selected=None):
        _fields(spec, {'path', 'sha256'}, f'Artifact {name}')
        if not isinstance(spec['path'], str) or not spec['path']:
            raise ValueError(f'Invalid artifact path: {name}')
        artifact = (path.parent / spec['path']).resolve(strict=True)
        if selected is not None and artifact != Path(selected).resolve(strict=True):
            raise ValueError(f'Generation-record {name} path differs from selected artifact')
        digest = _hash(artifact)
        if spec['sha256'] != digest:
            raise ValueError(f'Stale generation-record {name} hash: {artifact}')
        artifacts[name] = dict(path=str(artifact), sha256=digest, bytes=artifact.stat().st_size)

    for name, selected in actual.items():
        spec = record['artifacts'][name]
        if selected is None:
            if spec is not None:
                raise ValueError(f'Generation-record {name} is not part of scored inputs')
        else:
            bind(name, spec, selected)
    bind('checkpoint', record['checkpoint'])
    if not isinstance(record['code'], list) or not record['code']:
        raise ValueError('Generation record requires code artifacts')
    for index, spec in enumerate(record['code']):
        bind(f'code:{index}', spec)
    code_paths = [artifacts[f'code:{i}']['path'] for i in range(len(record['code']))]
    if len(code_paths) != len(set(code_paths)):
        raise ValueError('Duplicate generation code artifact')
    artifacts['record'] = dict(path=str(path), sha256=_hash(path), bytes=path.stat().st_size)
    evidence = dict(level='generation_record_binding', verified_input_use=False,
                    model=context, input_keys=[list(k) for k in expected_keys], artifacts=artifacts,
                    meaning='Current files match a supplied generation record; historical generation is not authenticated')
    if recompute is not None:
        if not callable(recompute) or expected_predictions is None:
            raise ValueError('A trusted callback and scored numeric predictions are required')
        expected = _numeric(copy.deepcopy(expected_predictions))
        actual_values = _numeric(recompute(copy.deepcopy(record)))
        if not _equal_numeric(expected, actual_values):
            raise ValueError('Independent recomputation differs from scored predictions')
        for spec in artifacts.values():
            if _hash(spec['path']) != spec['sha256']:
                raise ValueError('Bound artifact changed during recomputation')
        evidence.update(level='independently_recomputed', prediction_matches_recomputation=True,
                        comparison='exact numeric arrays and complete keys',
                        verification_trust='Executed callback supplied by the application, never by record metadata')
    elif expected_predictions is not None:
        raise ValueError('Numeric verification requires an executed callback')
    return evidence

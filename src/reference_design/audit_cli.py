"""Small operation planner and executable reference-role demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import audit
from .input_binding import _pairs


def _read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Invalid JSON number {x}')))


def _publish(output, value):
    from .cli import _destination, _publish as publish
    output = Path(output)
    _destination(output)
    publish(output, {'AUDIT.json': json.dumps(value, indent=2, allow_nan=False).encode() + b'\n'})


def run_demo(output):
    """Execute small synthetic controls; this is not an empirical benchmark."""
    refs = {'B1': np.array([1., 2., 4.]), 'B2': np.array([2., 1., 3.]),
            'B3': np.array([3., 4., 1.])}
    observed = np.array([4., 3., 5.])
    scales = np.array([1., 2., 1.])
    base = dict(task='reference_effect', output_kind='state', input_conditioned=True,
                model_input_id='B1', prediction_reference_id='B1', observation_reference_id='B2',
                storage_encoding='state', storage_baseline_id=None)
    outputs = {key: .4 * values + np.array([2., 1., 3.]) for key, values in refs.items()}
    env = dict(references=refs, observed=observed, scales=scales,
               metric_reference=np.zeros(3), candidates=np.stack([observed, observed + 1., observed - 1.]))

    def score(native, contract):
        stored, payload = audit.encode_prediction(native, contract, refs)
        decoded = audit.decode_prediction(stored, contract, refs, payload)
        return audit.canonical_components(decoded, contract, **env), stored, payload

    before, stored, payload = score(outputs['B1'], base)
    representation = dict(base, storage_encoding='effect', storage_baseline_id='B3')
    encoded, _, _ = score(outputs['B1'], representation)
    common_before = dict(base, prediction_reference_id='B1', observation_reference_id='B1')
    common_after = dict(base, prediction_reference_id='B2', observation_reference_id='B2')
    common0, _, _ = score(outputs['B1'], common_before)
    common1, _, _ = score(outputs['B1'], common_after)
    changed = dict(base, model_input_id='B2')
    changed_result, _, _ = score(outputs['B2'], changed)
    fixed_before = dict(base, input_conditioned=False)
    fixed_after = dict(fixed_before, model_input_id='B2')
    fixed0, _, _ = score(outputs['B1'], fixed_before)
    fixed1, _, _ = score(outputs['B1'], fixed_after)
    native_before = dict(base, output_kind='native_effect', storage_encoding='effect')
    native_after = dict(native_before, model_input_id='B2')
    negative = []
    for name, callback in [
        ('stale_input_prediction', lambda: audit.decode_prediction(stored, changed, refs, payload)),
        ('changed_prediction_array', lambda: audit.decode_prediction(stored + 1., base, refs, payload)),
        ('misspelled_active_role', lambda: audit.validate_contract(dict(base, model_inpt_id='B2'))),
        ('missing_active_prediction_reference', lambda: audit.validate_contract(dict(base, prediction_reference_id=None))),
    ]:
        try:
            callback()
        except ValueError as error:
            negative.append(dict(control=name, outcome='REJECTED', reason=str(error)))
        else:
            raise AssertionError(f'Negative control was accepted: {name}')
    errors = dict(representation=float(np.max(np.abs(before['scores']-encoded['scores']))),
                  common_reference=float(np.max(np.abs(common0['scores']-common1['scores']))),
                  unused_input=float(np.max(np.abs(fixed0['scores']-fixed1['scores']))))
    if any(error > 1e-12 for error in errors.values()):
        raise AssertionError(f'Equivalence control failed: {errors}')
    displacement = float(np.linalg.norm(changed_result['native_prediction']-before['native_prediction']))
    if not displacement > 0:
        raise AssertionError('Input-conditioned synthetic output did not change')
    # Deliberately failing to decode a storage representation must be detectable.
    effect_stored, _ = audit.encode_prediction(outputs['B1'], representation, refs)
    wrong = audit.canonical_components(effect_stored, base, **env)
    incorrect_error = float(np.max(np.abs(wrong['scores'][:, 0]-before['scores'][:, 0])))
    if not incorrect_error > 0:
        raise AssertionError('Incorrect-representation negative control had no effect')
    result = dict(schema='reference_design.audit_demo.v1', status='PASS_EXECUTED_SYNTHETIC_OPERATION_AUDIT',
                  evidence_level='executed_synthetic_controls', empirical_benchmark=False,
                  planner_uses='declared dependencies; no scores or arrays',
                  plans={name: audit.plan_change(b, a) for name, b, a in [
                      ('representation', base, representation), ('common_reference', common_before, common_after),
                      ('changed_input', base, changed), ('unused_input', fixed_before, fixed_after),
                      ('native_effect_input', native_before, native_after)]},
                  performed=['encoded and decoded equivalent storage', 'rescored equal shared references',
                             'generated both input-conditioned toy outputs', 'reused an input-independent toy output',
                             'executed stale-role and corrupted-array negative controls'],
                  planned_only=['native-effect input selection: planner checked; not counted as executed scoring'],
                  max_score_errors=errors, input_prediction_displacement=displacement,
                  incorrect_representation_mse_change=incorrect_error, negative_controls=negative,
                  historical_generation_authenticated=False)
    _publish(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='reference-design audit', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan', help='Plan changes between two declared role contracts')
    plan.add_argument('before', type=Path); plan.add_argument('after', type=Path)
    plan.add_argument('--output', type=Path, required=True)
    demo = commands.add_parser('demo', help='Execute small synthetic invariance and failure controls')
    demo.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            result = dict(schema='reference_design.audit_plan.v1', status='DECLARED_OPERATION_PLAN',
                          evidence_level='declaration', executed=False,
                          plan=audit.plan_change(_read(args.before), _read(args.after)))
            _publish(args.output, result)
        else:
            result = run_demo(args.output)
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=result['status'], output=str(args.output)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

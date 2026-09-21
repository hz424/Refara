"""Calculate the reference-sharing identity and its scope using small arrays."""
from itertools import product
from pathlib import Path
import argparse
import json

import numpy as np
from reference_design import Prediction, score_references, state_to_effect


def recompute():
    controls = np.array([[-1.0], [0.0], [1.0]])
    cases = []
    for effect, state in [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]:
        result = score_references([0.0], controls, [
            Prediction('effect', [effect], 'effect'),
            Prediction('state', [state], 'state'),
        ])
        # An independent residual calculation scores each assignment first.
        margins = {'S': [], 'D': []}
        for obs, pred, model in product(range(3), repeat=3):
            group = 'S' if obs == pred == model else 'D' if len({obs, pred, model}) == 3 else None
            if group is not None:
                target = -float(controls[obs, 0])
                effect_error = (effect - target) ** 2
                state_error = (state - float(controls[pred, 0]) - target) ** 2
                margins[group].append(state_error - effect_error)
        d_s, d_d = (float(np.mean(margins[g])) for g in ('S', 'D'))
        separation = float(3 * np.mean((controls - controls.mean(axis=0)) ** 2))
        pair = result.pairwise[0]
        np.testing.assert_allclose([pair['d_S'], pair['d_D'], pair['V']], [d_s, d_d, separation], atol=1e-12)
        np.testing.assert_allclose(d_d, d_s + separation, atol=1e-12)
        cases.append(dict(effect=effect, state=state, d_S=d_s, d_D=d_d, V=separation,
                          reversal=bool(d_s * d_d < 0)))

    states = np.array([[1., 3., -2.], [2., 0., 4.], [-1., 5., 2.]])
    treated = np.array([0.3, 1., -0.7])
    baseline, scales = np.array([2., -1., 0.5]), np.array([0.5, 2., 3.])
    residual = (states - treated) / scales
    recoded = (state_to_effect(states, baseline) - state_to_effect(treated, baseline)) / scales
    np.testing.assert_allclose(residual, recoded, atol=1e-12)

    # A native effect that changes with input falls outside the fixed-effect identity.
    input_controls = np.array([[0.], [1.], [2.]])
    counter = score_references([0.], input_controls, [
        Prediction('effect', input_controls, 'effect'), Prediction('state', [0.], 'state'),
    ]).pairwise[0]
    assert not counter['identity_applicable']
    assert not np.isclose(counter['d_D'] - counter['d_S'], counter['V'])
    return dict(
        status='PASS_REFERENCE_ROLE_MECHANISM', source_level='constructed arrays; direct residual enumeration',
        identity_scope='Three equal-depth blocks, balanced assignments, score before averaging, fixed direct effects and scales',
        cases=cases, equivalent_representation_max_abs=float(np.max(np.abs(residual - recoded))),
        input_dependent_effect_counterexample={k: counter[k] for k in ('d_S', 'd_D', 'V', 'identity_applicable')},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists() or output.is_symlink():
        parser.error('Use a new output directory')
    result = recompute()
    output.mkdir(parents=True)
    (output / 'REPLAY.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'])


if __name__ == '__main__':
    main()

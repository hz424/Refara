#!/usr/bin/env python3
"""Rebuild all aggregate Norman tables from the compact scored utility array."""
import argparse
from pathlib import Path

import numpy as np

from norman_reference import (DEPTHS, MODELS, PATTERNS, PATTERN_INDEX, TUPLES,
                              paired_tables, sha256, summarize_outputs, write_json)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=np.load(args.input,allow_pickle=False)
    if tuple(data['models'])!=MODELS or tuple(data['patterns'])!=PATTERNS or tuple(data['depths'])!=DEPTHS:
        raise ValueError('Stored axes differ from the frozen analysis')
    np.testing.assert_array_equal(data['role_tuples'],TUPLES)
    utility=data['utility']
    atomic=data['atomic_utility']
    pattern_recomputed=np.stack([atomic[...,ids].mean(axis=-1) for ids in PATTERN_INDEX.values()],axis=-1)
    np.testing.assert_allclose(utility,pattern_recomputed,rtol=0,atol=1e-12)
    us,um,up,uo,ud=[utility[...,:3,i] for i in range(5)]
    v=data['V'][None,...,None]
    k=data['K']
    errors={
        'atomic_to_pattern_max_abs':float(np.max(np.abs(utility-pattern_recomputed))),
        'M_minus_S_max_abs':float(np.max(np.abs(um-us))),
        'D_minus_S_plus_V_max_abs':float(np.max(np.abs(ud-us+v))),
        'P_minus_S_plus_V_plus_2K_max_abs':float(np.max(np.abs(up-us+v+2*k))),
        'O_minus_S_plus_V_minus_2K_max_abs':float(np.max(np.abs(uo-us+v-2*k))),
        'direct_zero_pattern_spread_max_abs':float(np.max(np.ptp(utility[...,3,:],axis=-1))),
    }
    if max(errors.values())>1e-9:
        raise AssertionError(errors)
    args.output.mkdir(parents=True,exist_ok=True)
    tasks=data['tasks'].tolist()
    margins=paired_tables(utility,tasks)
    summarize_outputs(args.output,utility,k,data['conditioning_state_distance_squared'],
                      data['V'],tasks,margins)
    write_json(args.output/'aggregate_replay_report.json',dict(
        status='PASS',input_sha256=sha256(args.input),identities=errors,
        script_sha256=sha256(__file__),
        outputs_sha256={p.name:sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}))


if __name__=='__main__':
    main()

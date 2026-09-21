#!/usr/bin/env python3
"""Regenerate Figure 5b using the recovered original generator and random streams.

The frozen generator, numerical primitives and directional resolver are loaded
without edits. This portable entry replaces only author-machine runtime/path
checks; all data generation, seeding, resampling and inference are original.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
COUNTS_NAME = 'outer_replicates_any_null_direction_rejected'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def module(name):
    path = HERE / 'frozen_original' / (name + '.py')
    spec = importlib.util.spec_from_file_location('_hierarchical_null_' + name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result

def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))

def load_original():
    expected = json.loads((HERE/'FROZEN_SOURCE_SHA256.json').read_text())
    for name, sha in expected.items():
        if digest(HERE/'frozen_original'/name) != sha:
            raise RuntimeError('Frozen numerical source changed: ' + name)
    if np.__version__ != '1.26.4':
        raise RuntimeError('Exact replay requires NumPy 1.26.4; install requirements.txt')
    runner = module('run_comparator_v1')
    numerics = module('comparator_numerics_v1')
    core = module('general_directional_family_v3')
    spec = json.loads((HERE/'frozen_original/COMPARATOR_EXECUTABLE_SPEC_V1.json').read_text())
    ledger = read_rows(HERE/'frozen_original/COMPARATOR_CELL_LEDGER_V1.tsv')
    # C02 is the hierarchical global-null law displayed in Figure 5b.
    selected = [row for row in ledger if row['law_id'] == 'C02']
    assert [(r['cell_id'], int(r['root_count'])) for r in selected] == [
        ('CMP002',8),('CMP006',12),('CMP010',20),('CMP014',40)]
    return runner, numerics, core, spec, selected

def aggregate(output, reference):
    expected_rows = read_rows(reference)
    expected = {(int(r['root_count_R']), r['procedure_id']):r for r in expected_rows}
    if len(expected) != 20:
        raise RuntimeError('Reference must contain all 20 distinct Figure 5b keys')
    rows=[]
    for roots in (8,12,20,40):
        payload=json.loads((output/f'root_{roots:02d}_counts.json').read_text())
        cell=payload['cell']
        assert cell['root_count']==roots and cell['law_id']=='C02'
        assert cell['outer_counts']==dict(attempted_replicates=2500,successful_replicates=2500,failed_replicates=0)
        for procedure, record in cell['inferential_procedures'].items():
            prior=expected[(roots,procedure)]
            events=record['counts'][COUNTS_NAME]
            rows.append(dict(root_count_R=roots, procedure_order=int(prior['procedure_order']),
                procedure_id=procedure, events=events, trials=2500, estimate=events/2500,
                archived_events=int(prior['events']), exact_events_match=events==int(prior['events']),
                source_record_id=prior['source_record_id']))
    rows.sort(key=lambda r:(r['root_count_R'],r['procedure_order']))
    with (output/'FIGURE_5B_REGENERATED_EVENT_COUNTS.tsv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    receipt=dict(status='PASS' if all(r['exact_events_match'] for r in rows) else 'FAIL',
        complete_keys=20,exact_event_count_matches=sum(r['exact_events_match'] for r in rows),
        root_counts=[8,12,20,40],outer_replicates_per_root_count=2500,
        procedure_count=5,bootstrap_resamples=499,failed_replicates=0,
        archived_summary_sha256=digest(reference),
        numerical_sources=json.loads((HERE/'FROZEN_SOURCE_SHA256.json').read_text()))
    (output/'FIGURE_5B_REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']!='PASS':
        raise RuntimeError('Original-stream replay differs from archived event counts; inspect comparison table')
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--root-count',type=int,choices=(8,12,20,40))
    parser.add_argument('--aggregate-only',action='store_true')
    parser.add_argument('--verify-against',type=Path)
    args=parser.parse_args()
    output=args.output_dir.absolute();output.mkdir(parents=True,exist_ok=True)
    if not args.aggregate_only:
        runner,numerics,core,spec,selected=load_original()
        if args.root_count is not None:
            selected=[r for r in selected if int(r['root_count'])==args.root_count]
        for row in selected:
            roots=int(row['root_count']);path=output/f'root_{roots:02d}_counts.json'
            if path.exists():raise FileExistsError(path)
            started=time.monotonic()
            cell=runner.run_cell(row,spec,numerics,core)
            payload=dict(cell=cell,runtime=dict(python=platform.python_version(),numpy=np.__version__),
                elapsed_seconds=time.monotonic()-started,
                execution=dict(slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID')),
                seed_namespace=row['seed_namespace'],
                seed_formula='SHA256(namespace + NUL + cell_id + NUL + OUTER + NUL + six_digit_index + NUL + purpose); unsigned big-endian integer',
                outer_indices=[0,2499],stream_purposes=list(runner.STREAM_LABELS),
                frozen_source_sha256=json.loads((HERE/'FROZEN_SOURCE_SHA256.json').read_text()),
                portable_entry_sha256=digest(Path(__file__)))
            path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
            print(json.dumps(dict(root_count=roots,counts=cell['outer_counts'],elapsed_seconds=payload['elapsed_seconds'])),flush=True)
            if cell['outer_counts']['failed_replicates']:
                raise RuntimeError('Outer replicate failed without replacement')
    if args.verify_against:
        print(json.dumps(aggregate(output,args.verify_against),indent=2))
    elif args.aggregate_only:
        parser.error('--aggregate-only requires --verify-against')

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Independent R=2 bootstrap check and integrated original-grid regression."""
from __future__ import annotations
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from run_low_n import load, module, HERE


def main():
    runner, numerics, core, spec, rows = load()
    old_numerics = module('regression_old_numerics', HERE / 'frozen_original/comparator_numerics_v1.py')
    with (HERE / 'frozen_original/COMPARATOR_CELL_LEDGER_V1.tsv').open(newline='') as stream:
        old_rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['law_id'] == 'C02']
    for row in old_rows:
        old = runner.run_cell(row, spec, old_numerics, core, test_outer_replicates=4)
        new = runner.run_cell(row, spec, numerics, core, test_outer_replicates=4)
        assert old == new
        assert new['outer_counts']['failed_replicates'] == 0
    # C02 makes each root an independent exchangeable continuous four-model
    # vector, so all 4! within-root rank orders are equiprobable. At R=2, a
    # shared pair ordering places its mean contrast beyond the centered root
    # bootstrap support. Only fully reversed root orders have no such pair.
    row = next(r for r in rows if r['root_count'] == '2')
    law = spec['hierarchical_gaussian_law']['laws']['C02']
    assert law['mean_vector_M01_to_M04'] == [0.,0.,0.,0.]
    assert law['mask_is_identical_across_roots_and_models']
    concordant = 0
    same_order_bootstrap_rejections = 0
    opposite_order_bootstrap_rejections = 0
    event_count = 0
    for i in range(2500):
        rng = np.random.Generator(np.random.PCG64(runner._outer_seed(row,i,'DATA')))
        root_rows, _, _, means = runner._analysis_rows(row,law,rng)
        has_pair = any((root_rows[0,a]-root_rows[0,b])*(root_rows[1,a]-root_rows[1,b]) > 0 for a,b in itertools.combinations(range(4),2))
        bootstrap = runner._bootstrap_rejections(root_rows,means,np.random.Generator(np.random.PCG64(runner._outer_seed(row,i,'BOOT_ROOT'))),numerics)
        concordant += has_pair
        event_count += bool(bootstrap)
        same_order_bootstrap_rejections += has_pair and bool(bootstrap)
        opposite_order_bootstrap_rejections += (not has_pair) and bool(bootstrap)
        assert (not has_pair) or bootstrap
    actual = json.loads((HERE/'results/root_02_counts.json').read_text())['cell']['inferential_procedures'][runner.ROOT_BOOTSTRAP_PROCEDURE]['counts']['outer_replicates_any_null_direction_rejected']
    assert event_count == actual
    receipt=dict(status='PASS',original_grid_regression=dict(root_counts=[8,12,20,40],outer_replicates_each=4,
                 all_five_procedure_count_structures_identical=True),
                 r2_whole_root_bootstrap=dict(analytic_limit=23/24,analytic_derivation='Only one of 4! relative model orderings is fully reversed',
                 trials=2500,concordant_pair_events=int(concordant),bootstrap_events=int(event_count),
                 concordant_events_with_rejection=int(same_order_bootstrap_rejections),
                 reversed_order_events_with_rejection=int(opposite_order_bootstrap_rejections),
                 archived_bootstrap_event_match=True,
                 finite_bootstrap_note='499 draws with plus-one p value; the bound beyond support gives p=1/500, below 0.05/12.'))
    (HERE/'qa/EXTENSION_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()

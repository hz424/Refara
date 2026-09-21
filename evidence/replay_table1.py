#!/usr/bin/env python3
"""Rebuild the full saved-score role summaries, then verify every configuration displayed by current Table 1."""
from pathlib import Path
import argparse,csv,json,math,subprocess,sys
ROOT=Path(__file__).resolve().parent


def check_selection(summary,selection):
    with Path(summary).open(newline='') as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
    if len(rows)!=432:raise ValueError('Expected all 432 role contrasts')
    checked=[]
    for expected in selection['selected_rows']:
        keys=['role','depth','model','metric']
        take=[r for r in rows if all(r[k]==expected[k] for k in keys)]
        if len(take)!=1:raise ValueError('Incomplete or duplicate Table 1 configuration')
        row=take[0];a=float(row['mean_abs_task_mean_change']);b=float(expected['mean_abs_task_mean_change'])
        if not math.isfinite(a) or abs(a-b)>2e-12:raise ValueError('Current Table 1 value differs')
        if row['model']!='selected_CPA' and a!=0:raise ValueError('Fixed-output input control must be exactly zero')
        checked.append({k:row[k] for k in keys+['mean_abs_task_mean_change']})
    if len(checked)!=16:raise ValueError('Expected six CPA role/depth rows and ten fixed-output controls')
    return checked


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.expanduser().resolve()
    if out.exists() or out.is_relative_to(ROOT):raise ValueError('Use a new output directory outside evidence/')
    selection=json.loads((ROOT/'TABLE1_CURRENT_SELECTION.json').read_text())
    subprocess.run([sys.executable,'-B',str(ROOT/'table1/code/replay_bundle.py'),'--out',str(out)],check=True)
    checked=check_selection(out/'metric_summary/role_contrasts.tsv',selection)
    cpa={(r['role'],r['depth']):float(r['mean_abs_task_mean_change']) for r in checked if r['model']=='selected_CPA'}
    ratios={depth:{role:cpa[role,depth]/cpa['Cmodel',depth] for role in ('Cpred','Cobs')} for depth in ('8','135')}
    result=dict(status='PASS_CURRENT_TABLE1_SAVED_SCORE_REPLAY',origin_release=selection['origin_release'],source_level='full bundled saved task-score cube, plus cached-array operation replay',fresh_model_fitting=False,fresh_model_prediction=False,fresh_cell_level_scoring=False,statistic=selection['statistic'],definition=selection['definition'],configuration_scope=selection['original_fit_vs_refit'],verified_rows=checked,
                scoring_to_input_sensitivity_ratios=ratios,
                ratio_definition='Ratio of summarized mean absolute task-averaged score changes; not the average of reassignment-wise ratios.',
                all_three_cpa_sensitivities_smaller_at_higher_depth=all(cpa[role,'135']<cpa[role,'8'] for role in ('Cmodel','Cpred','Cobs')))
    (out/'CURRENT_TABLE1.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'])

if __name__=='__main__':main()

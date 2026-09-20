#!/usr/bin/env python3
"""Build the plotted Norman summaries and the Extended Data Figure 8 workbook."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


DEPTHS = (8, 16, 32, 64, 128, 135)
MODELS = ('additive', 'compositional_ridge', 'CPA_0.8.8', 'zero_effect')
PATTERNS = ('S', 'M', 'P', 'O', 'D')
TIE_TOL = 1e-12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_xlsx(book, destination):
    raw = destination.with_suffix('.temporary.xlsx')
    book.save(raw)
    with zipfile.ZipFile(raw) as archive, zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as output:
        for name in sorted(archive.namelist()):
            info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            content = archive.read(name)
            if name == 'docProps/core.xml':
                import re
                content = re.sub(rb'(<dcterms:modified[^>]*>).*?(</dcterms:modified>)',
                                 rb'\g<1>2000-01-01T00:00:00Z\g<2>', content)
            output.writestr(info, content)
    raw.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads((args.results_dir / 'score_report.json').read_text())
    assert record['status'] == 'PASS'
    for filename, digest in record['output_sha256'].items():
        assert sha256(args.results_dir / filename) == digest, filename
    arrays = np.load(args.results_dir / 'utilities.npz', allow_pickle=False)
    assert tuple(arrays['depths']) == DEPTHS
    assert tuple(arrays['models']) == MODELS
    assert tuple(arrays['patterns']) == PATTERNS
    utility = arrays['utility']
    tasks = arrays['tasks'].tolist()
    assert utility.shape == (55, 30, 6, 4, 5)
    assert np.isfinite(utility).all()

    a_rows = []
    task_mean = utility.mean(axis=0)
    for label, di, mi, pi in itertools.product(range(30), range(6), range(4), range(5)):
        a_rows.append((label, DEPTHS[di], DEPTHS[di] * 8, MODELS[mi], PATTERNS[pi],
                       -task_mean[label, di, mi, pi]))
    a = pd.DataFrame(a_rows, columns=['allocation', 'depth_per_gemgroup', 'cells_per_reference',
                                     'model', 'pattern', 'scaled_squared_error'])
    a_summary = a.groupby(['depth_per_gemgroup', 'cells_per_reference', 'model', 'pattern'], sort=False).scaled_squared_error.agg(
        mean='mean', allocation_min='min', allocation_max='max').reset_index()
    existing = pd.read_csv(args.results_dir / 'depth_utility_summary.tsv', sep='\t')
    merged = a_summary.merge(existing, on=['depth_per_gemgroup', 'model', 'pattern'], validate='one_to_one')
    assert np.allclose(merged['mean'], -merged.mean_utility, rtol=0, atol=2e-15)
    assert np.allclose(merged.allocation_min, -merged.allocation_max_utility, rtol=0, atol=2e-15)
    assert np.allclose(merged.allocation_max, -merged.allocation_min_utility, rtol=0, atol=2e-15)

    distance = arrays['conditioning_state_distance_squared']
    assert distance.shape == (55, 30, 6, 3)
    assert np.max(np.abs(distance[:, :, :, :2])) == 0
    assert np.min(distance) >= 0
    # The square root follows the mean over tasks, genes and ordered block pairs.
    rms = np.sqrt(distance[:, :, :, 2].mean(axis=0))
    b = pd.DataFrame([(label, DEPTHS[di], DEPTHS[di] * 8, rms[label, di])
                       for label, di in itertools.product(range(30), range(6))],
                     columns=['allocation', 'depth_per_gemgroup', 'cells_per_reference', 'scaled_rms_displacement'])
    b_summary = b.groupby(['depth_per_gemgroup', 'cells_per_reference'], sort=False).scaled_rms_displacement.agg(
        mean='mean', allocation_min='min', allocation_max='max').reset_index()

    c_rows, averaged_rows = [], []
    for di, pi in itertools.product(range(6), range(5)):
        counts = np.zeros(30, dtype=int)
        averaged_count = 0
        for ma, mb in itertools.combinations(range(3), 2):
            anchor = utility[:, :, di, ma, 0] - utility[:, :, di, mb, 0]
            target = utility[:, :, di, ma, pi] - utility[:, :, di, mb, pi]
            crossing = (anchor * target < 0) & (np.abs(anchor) > TIE_TOL) & (np.abs(target) > TIE_TOL)
            counts += crossing.sum(axis=0)
            sa, ta = anchor.mean(axis=1), target.mean(axis=1)
            averaged_count += int(((sa * ta < 0) & (np.abs(sa) > TIE_TOL) & (np.abs(ta) > TIE_TOL)).sum())
        for label, count in enumerate(counts):
            c_rows.append((label, DEPTHS[di], DEPTHS[di] * 8, PATTERNS[pi], int(count), 165))
        averaged_rows.append((DEPTHS[di], DEPTHS[di] * 8, PATTERNS[pi], averaged_count, 165))
    c = pd.DataFrame(c_rows, columns=['allocation', 'depth_per_gemgroup', 'cells_per_reference',
                                    'pattern', 'reversed_task_pairs', 'eligible_task_pairs'])
    c_summary = c.groupby(['depth_per_gemgroup', 'cells_per_reference', 'pattern'], sort=False).reversed_task_pairs.agg(
        mean='mean', allocation_min='min', allocation_max='max').reset_index()
    after_mean = pd.DataFrame(averaged_rows, columns=['depth_per_gemgroup', 'cells_per_reference',
                                                     'pattern', 'reversed_after_allocation_mean', 'eligible_task_pairs'])
    margins = pd.read_csv(args.results_dir / 'task_depth_pair_summary.tsv', sep='\t')
    retained_primary = margins[margins.pair_role == 'primary_state_pair']
    totals = retained_primary.groupby(['depth_per_gemgroup', 'pattern']).strict_crossing_after_allocation_mean.sum()
    for row in after_mean.itertuples():
        assert row.reversed_after_allocation_mean == totals.loc[row.depth_per_gemgroup, row.pattern]
    assert not c.loc[c.pattern.isin(['S', 'M', 'D']), 'reversed_task_pairs'].any()

    tables = {
        'a_summary': a_summary, 'a_allocations': a,
        'b_summary': b_summary, 'b_allocations': b,
        'c_summary': c_summary, 'c_allocations': c,
        'crossings_after_mean': after_mean,
        'all_task_pair_summary': margins,
        'all_depth_pair_summary': pd.read_csv(args.results_dir / 'depth_pair_summary.tsv', sep='\t'),
    }
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / f'{name}.tsv', sep='\t', index=False, float_format='%.17g')
    notes = [
        ('Figure', 'Extended Data Figure 8: reference sensitivity in held-out genetic combinations'),
        ('Scope', 'One pooled Norman screen; eight equally weighted technical capture groups; no independent-experiment significance tests.'),
        ('Tasks', 'All 55 frozen held-out two-gene combinations; all 105 single-gene and 55 other combination conditions used for training.'),
        ('Primary models', 'Additive, compositional ridge and CPA are native-state predictors. Zero effect is a diagnostic baseline.'),
        ('Controls', '30 frozen allocations; cells per reference are 8 times the per-gemgroup depth.'),
        ('a', 'Scaled squared error is minus utility. Average all 55 task scores within each allocation, then report the mean and full range across allocations.'),
        ('a display', 'S and D are shown; every model and all five role designs are retained in a_summary and a_allocations.'),
        ('b', 'For each allocation, take the square root after averaging squared CPA prediction differences over tasks, genes and the six ordered pairs of input blocks. Then summarize the 30 allocations.'),
        ('b fixed predictors', 'The two predictors with fixed outputs have exactly zero conditioning displacement and are not drawn as duplicate zero series.'),
        ('c', 'For each allocation, count strict sign changes relative to S among all 165 primary model-pair/task comparisons. Then summarize counts across the 30 allocations.'),
        ('c display', 'P and O are shown; S, M and D have exactly zero crossings and remain in the source tables.'),
        ('Separate aggregation', 'crossings_after_mean first averages each task/model/design score across allocations and then counts crossings. This is a different calculation from panel c.'),
        ('Roles', 'S: all roles share one block; M: observation and prediction share; P: observation and model input share; O: prediction and model input share; D: all three distinct.'),
        ('Tie policy', 'Margins with absolute value at most 1e-12 are numerical ties, not strict crossings.'),
        ('Variability', 'Minima and maxima are finite ranges over control allocations, not confidence intervals. Tasks and allocations are not biological replicates.'),
        ('Complete comparisons', 'all_task_pair_summary and all_depth_pair_summary preserve all six model pairs, including the separately identified zero-effect diagnostics.'),
        ('Score definition', 'Mean squared effect residual over 2000 genes, scaled by fixed TRAIN-control standard deviations floored at 0.1; matched scoring and aggregation follow the frozen protocol.'),
    ]
    book = Workbook()
    book.remove(book.active)
    book.properties.creator = 'Reference-cell benchmark analysis'
    book.properties.title = 'Extended Data Figure 8 source data'
    book.properties.created = datetime(2000, 1, 1)
    book.properties.modified = datetime(2000, 1, 1)
    tables_for_book = {'Read me': pd.DataFrame(notes, columns=['Item', 'Description']), **tables}
    for name, frame in tables_for_book.items():
        sheet = book.create_sheet(name)
        sheet.append(frame.columns.tolist())
        for row in frame.itertuples(index=False, name=None):
            sheet.append([value.item() if isinstance(value, np.generic) else value for value in row])
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='3D7180')
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = min(42, max(14, len(str(column[0].value)) + 2))
    stable_xlsx(book, args.output_dir / 'Extended_Data_Figure_8_Source_Data.xlsx')
    receipt = dict(status='PASS', utility_shape=list(utility.shape), biological_experiments=1,
                   source_record_sha256=sha256(args.results_dir / 'score_report.json'),
                   source_utilities_sha256=sha256(args.results_dir / 'utilities.npz'),
                   tables={name: len(frame) for name, frame in tables.items()},
                   output_sha256={path.name: sha256(path) for path in sorted(args.output_dir.iterdir())
                                  if path.suffix in ('.tsv', '.xlsx')})
    (args.output_dir / 'SOURCE_DATA_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

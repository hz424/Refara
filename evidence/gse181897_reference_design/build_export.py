"""Export an authenticated local execution as a portable numerical capsule.

All local input paths are supplied explicitly. Only compact task losses and
metadata are read; raw assay matrices, model weights and PyTorch are unnecessary.
The output describes a local pre-outcome freeze, not a public preregistration.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import re
import shutil
import tempfile

import replay
from paired_statistics import EXECUTED_SOURCE_SHA256


CODE_FILES = ('replay.py', 'paired_statistics.py', 'build_export.py')
TASK_FIELDS = ('donor', 'partition', 'context', 'condition', 'secondary_32_eligible',
               'control_cells_available', 'within_donor_weight')
SELECTION_FIELDS = ('donor', 'context', 'condition', 'budget', 'rule', 'family', 'loss', 'seeds')
ASSESSMENT_FIELDS = ('donor', 'context', 'condition', 'deployment_n', 'family', 'loss',
                     'treated_state_loss', 'seeds', 'control_pool', 'treated_pool')


def authenticated(record):
    path = Path(record['path'])
    replay.require(path.is_file() and replay.sha256(path) == record['sha256'], 'Internal source artifact hash differs')
    return path


def execution_stage(directory, protocol_sha, expected_stage):
    path = Path(directory) / 'receipt.json'
    receipt = json.loads(path.read_text())
    replay.require(receipt['stage'] == expected_stage and receipt['status'] == 'complete' and
                   receipt['protocol_sha256'] == protocol_sha, 'Internal execution stage is not complete or belongs to another protocol')
    for record in receipt['files'].values():
        authenticated(record)
    for record in receipt['inputs'].values():
        authenticated(record)
    return receipt


def write_gzip(path, rows, fields):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)


def recode_rows(rows, fields, donor_map, pool_map=None):
    output = []
    for row in rows:
        item = {key: row[key] for key in fields}
        item['donor'] = donor_map[item['donor']]
        if pool_map is not None:
            for key in ('control_pool', 'treated_pool'):
                item[key] = pool_map[item[key]]
        output.append(item)
    return output


def recode_statistics(value, donor_map):
    if isinstance(value, dict):
        return {key: donor_map[item] if key in ('unit_id', 'omitted_unit') else recode_statistics(item, donor_map)
                for key, item in value.items()}
    if isinstance(value, list):
        return [recode_statistics(item, donor_map) for item in value]
    return value


def typed_table(rows):
    integers = {'budget', 'deployment_n', 'donors', 'tasks', 'benchmark_input_cells_per_fold'}
    floats = {'mean_shared_loss', 'mean_rule_loss', 'relative_loss_reduction', 'mean_shared_treated_state_loss',
              'mean_rule_treated_state_loss', 'treated_state_relative_loss_reduction', 'uncached_inference_ratio',
              'mean_overlap_selected_loss', 'mean_disjoint_selected_loss', 'disjoint_loss_reduction_vs_overlap'}
    output = []
    for row in rows:
        output.append({key: int(value) if key in integers else float(value) if key in floats and value != ''
                       else value == 'True' if key == 'confirmatory_primary' else value for key, value in row.items()})
    return output


def model_contract(internal, root):
    inventory = json.loads(authenticated(internal['files']['model_inventory']).read_text())
    adapter = json.loads(authenticated(internal['files']['adapter_spec']).read_text())
    download_path = root / 'data' / 'download_aggr_manifest.json'
    download = json.loads(download_path.read_text())
    replay.require(download['uncompressed']['sha256'] == internal['files']['raw_aggr']['sha256'],
                   'GEO download receipt and frozen assay digest differ')
    def digest_record(record):
        return {key: record[key] for key in ('sha256', 'bytes') if key in record}
    def fit_record(entry):
        result = {key: entry[key] for key in ('family', 'seed', 'candidate_id', 'native_output')}
        result.update({'checkpoint': digest_record(entry['model']),
                       'completion': digest_record(entry['completion']),
                       'dependencies': [dict(name=Path(record['path']).name, **digest_record(record))
                                        for record in entry.get('dependencies', [])]})
        for key in ('hyperparameters', 'parameters', 'architecture', 'training_configuration'):
            if key in entry:
                result[key] = entry[key]
        return result
    contract = {'schema': 'REFARA_FIXED_MODEL_CONTRACT_V1',
                'inventory_sha256': internal['files']['model_inventory']['sha256'],
                'admitted_fits': [fit_record(entry) for entry in inventory['models'] if entry['family'] in internal['models']],
                'excluded_fits': [fit_record(entry) for entry in inventory['models'] if entry['family'] not in internal['models']],
                'excluded_families': adapter['excluded_families'],
                'original_hyperparameter_selection': inventory['selection'],
                'hyperparameter_scope': 'Original fixed candidate identifiers and any explicit inventory hyperparameters are recorded. No model weights are retrained or selected using external assessment outcomes.',
                'native_input_feature_count': adapter['input_panel_size'], 'score_feature_count': adapter['score_panel_size'],
                'imputed_input_feature_count': len(adapter['masked_indices']),
                'training_panel_axis': digest_record(inventory['panel_indices']),
                'training_full_feature_axis': digest_record(inventory['source_features']),
                'training_native_scales': digest_record(inventory['scales']),
                'scoring_scales': digest_record(adapter['scoring_scales']),
                'imputation_means': digest_record(adapter['imputation_means']),
                'gene_mapping': digest_record(adapter['mapping']),
                'score_gene_symbols': adapter['score_gene_symbols'],
                'excluded_score_genes': adapter['excluded_score_genes'],
                'input_normalization': adapter['input_normalization'],
                'imputation': adapter['imputation'], 'feature_weights': adapter['feature_weights'],
                'training_control_support': {key: adapter['training_source'][key]
                                             for key in ('controls_per_context', 'controls_per_donor_context')},
                'training_donor_count': len(adapter['training_source']['donors']),
                'technical_admission': adapter['technical_admission'],
                'fidelity_is_admission_threshold': adapter['fidelity_is_admission_threshold'],
                'native_output_semantics': inventory['state_effect_semantics'],
                'seed_semantics': inventory['seed_semantics'],
                'training_code': [dict(name=Path(record['path']).name, **digest_record(record))
                                  for record in inventory['source_code']],
                'geo_source': {'accession': 'GSE181897', 'geo_url': download['geo_url'],
                               'raw_counts_url': download['source_url'],
                               'retrieved_utc': download['retrieved_utc'],
                               'raw_counts': digest_record(internal['files']['raw_aggr']),
                               'download_receipt_sha256': replay.sha256(download_path)},
                'availability': 'Checkpoint digests identify the executed fixed models. This compact capsule contains derived task losses and does not bundle checkpoints, raw expression, imputation arrays or other private files.'}
    # Any future inventory hyperparameter additions must not leak machine paths.
    text = json.dumps(contract)
    replay.require(re.search(r'/(?:dpc|home|tmp)/', text) is None, 'Model contract contains a local path')
    return contract


def export(source_dir, protocol_path, selection_dir, assessment_dir, report_dir, output, *, freeze_time_utc=None):
    root = Path(source_dir).resolve()
    def source(path):
        path = Path(path)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        replay.require(resolved.is_relative_to(root), 'Execution input must be inside --source-dir')
        return resolved
    protocol_path = source(protocol_path)
    protocol_sha = replay.sha256(protocol_path)
    internal = json.loads(protocol_path.read_text())
    replay.require(internal['status'] == 'frozen_before_external_expression', 'Internal protocol lacks pre-outcome freeze')
    replay.require(internal['code']['decision_analysis.py']['sha256'] == EXECUTED_SOURCE_SHA256,
                   'Public statistics do not match the frozen executed source')
    for record in internal['code'].values():
        authenticated(record)
    selection = execution_stage(source(selection_dir), protocol_sha, 'selection')
    assessment = execution_stage(source(assessment_dir), protocol_sha, 'assessment')
    report = execution_stage(source(report_dir), protocol_sha, 'report')
    decisions_path = authenticated(selection['files']['decisions'])
    replay.require(assessment['inputs']['decisions']['sha256'] == replay.sha256(decisions_path) and
                   report['inputs']['decisions']['sha256'] == replay.sha256(decisions_path), 'Assessment/report decision binding differs')
    decisions = json.loads(decisions_path.read_text())
    summary = json.loads(authenticated(report['files']['summary']).read_text())
    original_tasks = replay.read_rows(authenticated(internal['files']['eligible_tasks']))
    original_selection = replay.read_rows(authenticated(selection['files']['losses']))
    original_assessment = replay.read_rows(authenticated(assessment['files']['losses']))
    contract = model_contract(internal, root)
    donor_sets = [{r['donor'] for r in original_tasks if r['partition'] == partition}
                  for partition in ('selection', 'assessment')]
    replay.require(not donor_sets[0].intersection(donor_sets[1]), 'Donor partitions overlap')
    donor_map = {}
    for partition, prefix in (('selection', 'S'), ('assessment', 'A')):
        names = sorted({r['donor'] for r in original_tasks if r['partition'] == partition})
        donor_map.update({name: f'{prefix}{index + 1:03d}' for index, name in enumerate(names)})
    replay.require(len(donor_map) == len({r['donor'] for r in original_tasks}), 'Donor partitions overlap')
    pools = sorted({r[key] for r in original_assessment for key in ('control_pool', 'treated_pool')})
    pool_map = {name: f'P{index + 1:02d}' for index, name in enumerate(pools)}
    tasks = recode_rows(original_tasks, TASK_FIELDS, donor_map)
    selection_rows = recode_rows(original_selection, SELECTION_FIELDS, donor_map)
    assessment_rows = recode_rows(original_assessment, ASSESSMENT_FIELDS, donor_map, pool_map)
    freeze = freeze_time_utc or internal.get('frozen_utc') or internal.get('created_utc')
    replay.require(bool(freeze), 'Supply the actual local freeze UTC time; do not infer a public preregistration')
    instant = datetime.fromisoformat(freeze.replace('Z', '+00:00'))
    replay.require(instant.tzinfo is not None and instant.utcoffset() == timezone.utc.utcoffset(instant), 'Freeze time must explicitly be UTC')
    portable = {key: internal[key] for key in ('models', 'primary_budget', 'secondary_budgets', 'recommended_rule',
                'primary_deployment_n', 'secondary_deployment_n', 'confidence', 'bootstrap_seed',
                'bootstrap_resamples', 'minimum_relative_reduction', 'max_inference_ratio')}
    portable.update({'schema': replay.SCHEMA, 'conditions': list(replay.CONDITIONS),
                     'tie_tolerance': decisions['tie_tolerance'],
                     'minimum_inference_donors': internal.get('minimum_inference_donors', 24),
                     'source_frozen_protocol_sha256': protocol_sha,
                     'statistics_executed_source_sha256': EXECUTED_SOURCE_SHA256,
                     'executed_runner_sha256': internal['code']['run_external.py']['sha256'],
                     'local_freeze': {'time_utc': freeze, 'scope': 'Local protocol and code were frozen before external outcome analysis; this is not a claim of public preregistration.'},
                     'source': {'accession': 'GSE181897', 'url': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE181897'},
                     'inference_scope': summary['scope'], 'primary_endpoint': summary['endpoint'],
                     'treated_state_secondary': summary['treated_state_secondary'],
                     'eligibility': 'At least 16 controls and 8 treated cells for each of three conditions in every eligible donor/context; metadata-only complete condition blocks.',
                     'weighting': 'Equal eligible donors, equal eligible contexts within donor, equal three conditions. Secondary subsets renormalize these prespecified within-donor weights.',
                     'adapter': 'Fixed training-control mean imputation of 32 unavailable inputs; 1968 unambiguous genes scored with original training scales and uniform weights. Seven fixed families; CPA excluded before outcomes.',
                     'freeze_scope': internal.get('freeze_scope', 'Locally frozen before external outcomes; not publicly preregistered.'),
                     'numerical_verification': {'absolute_tolerance': replay.EXPECTED_ATOL,
                                                'relative_tolerance': replay.EXPECTED_RTOL,
                                                'original_inference_scipy': '1.12.0',
                                                'observed_t_interval_difference_with_scipy_1_15_3': 2.457921019494158e-13,
                                                'reason': 'SciPy Student-t quantile implementations differ in final digits. Original frozen expected values are retained; selection labels and joint decisions must match exactly, and the benefit boundary remains strictly zero.'},
                     'donor_relabeling': 'Partition-specific monotone pseudonyms preserve the original donor sort order, including the fixed bootstrap draw order. No original cell identities are exported.',
                     'replay_scope': 'Independent numerical recomputation from saved per-task model losses. This capsule does not rerun expression preprocessing or neural model inference.',
                     'internal_receipt_sha256': {'selection': replay.sha256(source(selection_dir) / 'receipt.json'),
                                                  'assessment': replay.sha256(source(assessment_dir) / 'receipt.json'),
                                                  'report': replay.sha256(source(report_dir) / 'receipt.json')}})
    expected = {key: summary[key] for key in ('recommended_rule', 'joint_practical_benefit', 'joint_adoption')}
    expected['paired_t'] = recode_statistics(summary['paired_t'], donor_map)
    expected['studentized_bootstrap'] = recode_statistics(summary['studentized_bootstrap'], donor_map)
    expected['selection_decisions'] = {key: decisions[key] for key in ('families', 'recommended_rule', 'tie_tolerance', 'budgets')}
    expected['policy_summary'] = typed_table(replay.read_rows(authenticated(report['files']['policy_summary'])))
    expected['donor_losses'] = expected['paired_t']['units']
    expected['mechanism_diagnostics'] = typed_table(replay.read_rows(authenticated(report['files']['diagnostics'])))
    expected['compute'] = {key: summary['compute'][key] for key in ('shared_work', 'recommended_work', 'ratio')}
    expected['all_model_losses'] = []
    for n in (8, 16):
        for family in replay.FAMILIES:
            rows = [r for r in original_assessment if int(r['deployment_n']) == n and r['family'] == family]
            effect, donors = replay.mean_by_donor(rows)
            state, _ = replay.mean_by_donor(rows, 'treated_state_loss')
            expected['all_model_losses'].append({'deployment_n': n, 'family': family, 'donors': donors,
                                                 'tasks': len(rows), 'mean_effect_loss': effect,
                                                 'mean_treated_state_loss': state})
    # Independently recompute all saved choices and statistics before publishing.
    recomputed = replay.analyse(portable, tasks, selection_rows, assessment_rows)
    replay.compare_expected(replay.expected_projection(recomputed), expected)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay.require(not (output / 'protocol.json').exists() and not (output / 'manifest.json').exists(), 'Capsule data already exported')
    replay.require(not (output / 'source_data').exists() or not any((output / 'source_data').iterdir()), 'Capsule source_data is not empty')
    with tempfile.TemporaryDirectory(prefix='.reference-export-', dir=output.parent) as temp:
        temp = Path(temp)
        (temp / 'source_data').mkdir()
        for name in CODE_FILES:
            current = Path(__file__).resolve().parent / name
            if (output / name).exists():
                replay.require(replay.sha256(output / name) == replay.sha256(current), 'Existing capsule code differs')
            shutil.copyfile(current, temp / name)
        replay.write_json(temp / 'model_contract.json', contract)
        portable['model_contract'] = {'path': 'model_contract.json', 'sha256': replay.sha256(temp / 'model_contract.json')}
        replay.write_json(temp / 'protocol.json', portable)
        replay.write_json(temp / 'source_data/expected.json', expected)
        write_gzip(temp / 'source_data/tasks.tsv.gz', tasks, TASK_FIELDS)
        write_gzip(temp / 'source_data/selection_losses.tsv.gz', selection_rows, SELECTION_FIELDS)
        write_gzip(temp / 'source_data/assessment_losses.tsv.gz', assessment_rows, ASSESSMENT_FIELDS)
        names = list(CODE_FILES) + ['protocol.json', 'model_contract.json', 'source_data/expected.json', 'source_data/tasks.tsv.gz',
                                 'source_data/selection_losses.tsv.gz', 'source_data/assessment_losses.tsv.gz']
        manifest = {'schema': 'REFARA_CAPSULE_MANIFEST_V1', 'files': {name: replay.sha256(temp / name) for name in names}}
        replay.write_json(temp / 'manifest.json', manifest)
        replay.load_capsule(temp)  # Check the fully portable artifact and digests.
        (output / 'source_data').mkdir(exist_ok=True)
        for name in names + ['manifest.json']:
            shutil.copyfile(temp / name, output / name)
    return {'source_frozen_protocol_sha256': protocol_sha, 'donors': len(donor_map),
            'selection_rows': len(selection_rows), 'assessment_rows': len(assessment_rows),
            'expected_result_verified': True, 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--freeze-time-utc')
    args = parser.parse_args(argv)
    print(json.dumps(export(args.source_dir, args.protocol, args.selection, args.assessment,
                            args.report, args.output, freeze_time_utc=args.freeze_time_utc)))


if __name__ == '__main__':
    main()

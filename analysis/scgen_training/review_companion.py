#!/usr/bin/env python3
"""Verify the confidential companion and recheck completed training evidence.

BSD-3-Clause; see LICENSE. No fitting, model loading or raw-data reconstruction.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

ARCHIVE_SHA = '8bd0b1545f3be887d86efb26993636b2b8f21193d5643a0f773f42b0be935fea'
ARCHIVE_BYTES = 1753488681
PREFIX = 'SCGEN_TRAINING_ROBUSTNESS_20260910_V1/'
MANIFEST_SHA = '969b39ed10ac9b982643de3933f4f371a3f8e4a2d2db7c55f7d5cfd39dd1e3dd'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def member_path(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name, 'Unsafe archive path')
    require(name.startswith(PREFIX), 'Unexpected archive root')
    require(not any(s in p.parts for s in ('__pycache__', 'PRIVATE_EXTRACTION_ONLY')), 'Unexpected private/cache directory')
    require(not name.endswith(('.pyc', '.pyo')), 'Unexpected compiled Python file')
    return name[len(PREFIX):]


def verify(archive):
    require(archive.is_file() and not archive.is_symlink(), 'Archive must be a regular file')
    if archive.stat().st_size != ARCHIVE_BYTES or file_sha(archive) != ARCHIVE_SHA:
        return verify_review_archive(archive)
    require(archive.stat().st_size == ARCHIVE_BYTES, 'Archive byte count differs')
    require(file_sha(archive) == ARCHIVE_SHA, 'Archive SHA-256 differs')
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        require(len(entries) == 534 and len({x.filename for x in entries}) == 534, 'Archive membership differs')
        names = set()
        for entry in entries:
            relative = member_path(entry.filename)
            require(not entry.is_dir(), 'Archive includes a directory entry')
            mode = entry.external_attr >> 16
            require(not stat.S_ISLNK(mode) and (stat.S_IFMT(mode) in (0, stat.S_IFREG)), 'Archive includes a nonregular member')
            names.add(relative)
        manifest_bytes = z.read(PREFIX + 'MANIFEST.sha256')
        require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, 'Manifest SHA-256 differs')
        manifest = {}
        for line in manifest_bytes.decode('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            require(name not in manifest and len(digest) == 64, 'Invalid manifest record')
            member_path(PREFIX + name)
            manifest[name] = digest
        require(set(manifest) == names - {'MANIFEST.sha256'}, 'Manifest does not cover every payload')
        for name, expected in manifest.items():
            h = hashlib.sha256()
            with z.open(PREFIX + name) as stream:
                for data in iter(lambda: stream.read(1 << 20), b''):
                    h.update(data)
            require(h.hexdigest() == expected, 'Archive member hash differs: ' + name)
        # Reading every member to EOF also verifies each ZIP CRC.
        prepared = sorted(n for n in names if n.startswith('05_TRAINING_RECORDS/prepared/') and n.endswith('/manifest.json'))
        cv = sorted(n for n in names if n.startswith('05_TRAINING_RECORDS/runs/cv/') and n.endswith('/FIT_RECEIPT.json'))
        refits = sorted(n for n in names if n.startswith('05_TRAINING_RECORDS/runs/refit/') and n.endswith('/FIT_RECEIPT.json'))
        cv_checkpoints = [n for n in names if n.startswith('05_TRAINING_RECORDS/runs/cv/') and n.endswith('.ckpt')]
        refit_checkpoints = [n for n in names if n.startswith('05_TRAINING_RECORDS/runs/refit/') and n.endswith('.ckpt')]
        require((len(prepared), len(cv), len(refits), len(cv_checkpoints), len(refit_checkpoints)) == (8, 14, 3, 42, 3), 'Training evidence family differs')
        return {'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES, 'members': len(entries), 'payload_hashes_and_crc_checked': len(manifest), 'prepared_bundles': len(prepared), 'cv_trajectories': len(cv), 'cv_optimizer_checkpoints': len(cv_checkpoints), 'final_refits': len(refits), 'final_optimizer_checkpoints': len(refit_checkpoints), 'model_objects_deserialized': False}


def verify_review_archive(archive, profile_path=None):
    """Verify an explicitly recorded review distribution and all of its members."""
    profile_path = profile_path or Path(__file__).with_name('review_archive.json')
    primary = json.loads(profile_path.read_text())
    profiles = [primary, *primary.get('alternate_distributions', [])]
    matching_size = [p for p in profiles if archive.stat().st_size == p['archive_bytes']]
    require(matching_size, 'Archive byte count differs')
    archive_sha = file_sha(archive)
    profile = next((p for p in matching_size if p['archive_sha256'] == archive_sha), None)
    require(profile is not None, 'Archive SHA-256 differs')
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        require(len(entries) == profile['members'] and len({x.filename for x in entries}) == len(entries),
                'Archive membership differs')
        names = set()
        for entry in entries:
            relative = member_path(entry.filename)
            mode = entry.external_attr >> 16
            require(not entry.is_dir() and not stat.S_ISLNK(mode)
                    and stat.S_IFMT(mode) in (0, stat.S_IFREG), 'Archive includes a nonregular member')
            names.add(relative)
        raw = z.read(PREFIX + 'MANIFEST.sha256')
        require(hashlib.sha256(raw).hexdigest() == profile['manifest_sha256'], 'Manifest SHA-256 differs')
        original = z.read(PREFIX + 'ORIGINAL_MANIFEST.sha256')
        require(hashlib.sha256(original).hexdigest() == MANIFEST_SHA, 'Original scientific manifest differs')
        manifest = {}
        for line in raw.decode('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            member_path(PREFIX + name)
            require(name not in manifest and len(digest) == 64, 'Invalid manifest record')
            manifest[name] = digest
        require(set(manifest) == names - {'MANIFEST.sha256'}, 'Manifest does not cover every payload')
        for name, expected in manifest.items():
            digest = hashlib.sha256()
            with z.open(PREFIX + name) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(chunk)
            require(digest.hexdigest() == expected, 'Archive member hash differs: ' + name)
    return {'archive_sha256': profile['archive_sha256'], 'archive_bytes': profile['archive_bytes'],
            'members': len(entries), 'payload_hashes_and_crc_checked': len(manifest),
            'original_scientific_manifest_sha256': MANIFEST_SHA,
            'model_objects_deserialized': False, 'distribution': 'review'}


def scientific_manifest(root):
    """Locate the original hash list used to verify scientific replay inputs."""
    original = root / 'ORIGINAL_MANIFEST.sha256'
    path = original if original.exists() else root / 'MANIFEST.sha256'
    require(path.is_file() and not path.is_symlink() and file_sha(path) == MANIFEST_SHA,
            'The fixed scientific manifest is required')
    return path


def extract_one(z, member, destination):
    member_path(PREFIX + member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with z.open(PREFIX + member) as src, destination.open('xb') as dst:
        for chunk in iter(lambda: src.read(1 << 20), b''):
            dst.write(chunk)
    destination.chmod(0o600)


def module_from(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def projected_selection(z, root, selector):
    """Reaggregate unchanged risks when protocol prose is a reviewer projection."""
    seal = json.loads((root / 'PROTOCOL_SEAL.json').read_text())
    require(seal.get('status') == 'SEALED_REVIEWER_PROJECTION'
            and seal.get('original_protocol_sha256') == selector.PROTOCOL_SHA
            and seal.get('protocol_sha256') == file_sha(root / 'PROTOCOL.json')
            and seal.get('reviewer_projection', {}).get('scientific_design_changed') is False,
            'Unrecognized protocol projection')
    original = z.read(PREFIX + 'ORIGINAL_MANIFEST.sha256')
    require(hashlib.sha256(original).hexdigest() == MANIFEST_SHA,
            'Original scientific manifest differs')
    hashes = {name: digest for digest, name in
              (line.split('  ', 1) for line in original.decode().splitlines())}

    def read_original(name):
        raw = z.read(PREFIX + name)
        require(name in hashes and hashlib.sha256(raw).hexdigest() == hashes[name],
                'Scientific selection input differs: ' + name)
        return json.loads(raw)

    pools = {(lr, epoch): [] for lr in selector.RATES for epoch in selector.EPOCHS}
    seen_donors = set()
    for fold in selector.FOLDS:
        manifest = read_original(f'05_TRAINING_RECORDS/prepared/{fold}/manifest.json')
        donors = set(manifest['val_donor_ids'])
        require(not seen_donors.intersection(donors), 'Repeated validation donor')
        seen_donors.update(donors)
        for rate in selector.RATES:
            run = f'05_TRAINING_RECORDS/runs/cv/{fold}_lr{rate}'
            receipt = read_original(run + '/FIT_RECEIPT.json')
            require(receipt['status'] == 'PASS_TRAINING_ONLY_CV'
                    and receipt['fold_id'] == fold and receipt['learning_rate'] == rate,
                    'CV identity differs')
            for epoch in selector.EPOCHS:
                diagnostic = read_original(run + f'/checkpoints/epoch{epoch:03}_validation.json')
                require(diagnostic['fold_id'] == fold and diagnostic['learning_rate'] == rate
                        and diagnostic['epoch'] == epoch, 'Diagnostic identity differs')
                pools[rate, epoch].extend(selector.validate_donor_rows(diagnostic, donors))
    require(seen_donors == set(range(40)), 'Validation donor coverage differs')
    candidates, chosen = selector.rank_complete_candidates([
        {'learning_rate': rate, 'epoch': epoch, 'per_donor': rows}
        for (rate, epoch), rows in pools.items()])
    return {'selected_learning_rate': chosen['learning_rate'], 'selected_epoch': chosen['epoch'],
            'selected_equal_donor_prediction_risk': chosen['equal_donor_prediction_risk'],
            'primary_risk_candidate_grid': candidates, 'complete_folds': 7,
            'complete_trajectories': 14, 'complete_candidates': 6, 'validation_donors': 40}


def selection_check(z, root):
    # A temporary view restores only the original selector's declared JSON layout.
    for name in ('PROTOCOL.json', 'PROTOCOL_SEAL.json', 'PROTOCOL_TECHNICAL_AMENDMENT_001.json', 'PREPARATION_RECEIPT.json'):
        extract_one(z, '06_AUDIT_RECORDS/' + name, root / name)
    for fold in [f'fold{i:02}' for i in range(1, 8)] + ['full']:
        extract_one(z, f'05_TRAINING_RECORDS/prepared/{fold}/manifest.json', root / 'prepared' / fold / 'manifest.json')
    for fold in (f'fold{i:02}' for i in range(1, 8)):
        for rate in ('0.001', '0.0003'):
            run = f'{fold}_lr{rate}'
            for suffix in ['FIT_RECEIPT.json'] + [f'checkpoints/epoch{epoch:03}_validation.json' for epoch in (80,160,320)]:
                extract_one(z, f'05_TRAINING_RECORDS/runs/cv/{run}/{suffix}', root / 'runs/cv' / run / suffix)
    script = root / 'select_training_configuration.py'
    extract_one(z, '04_CODE/select_training_configuration.py', script)
    selector = module_from(script, 'archived_training_selector')
    projected = file_sha(root / 'PROTOCOL.json') != selector.PROTOCOL_SHA
    actual = (projected_selection(z, root, selector) if projected
              else selector.collect_selection(root, root / 'runs/cv'))
    expected = json.loads(z.read(PREFIX + '06_AUDIT_RECORDS/TRAINING_SELECTION.json'))
    keys = ('selected_learning_rate', 'selected_epoch', 'selected_equal_donor_prediction_risk', 'primary_risk_candidate_grid', 'complete_folds', 'complete_trajectories', 'complete_candidates', 'validation_donors')
    require(all(actual[k] == expected[k] for k in keys), 'Review reaggregation differs from archived selection')
    return {'status': 'PASS_REVIEW_SELECTION_REAGGREGATION', **{k: actual[k] for k in keys}, 'archived_selector_sha256': file_sha(script), 'new_selection_or_chronology_seal_created': False, 'new_model_fitting': False, 'reviewer_protocol_projection': projected, 'input_scope': 'Archived fold manifests, CV fit receipts and held-out-training donor/task risks; no expression, checkpoint or external result was supplied to the selector.'}


def utility_check(z, root):
    for name in ('TRAINING_SELECTION.json', 'TRAINING_SELECTION_SEAL.json'):
        extract_one(z, '06_AUDIT_RECORDS/' + name, root / name)
    for name in z.namelist():
        relative = name[len(PREFIX):]
        start = '03_SOURCE_DATA/external_evaluation/'
        if relative.startswith(start):
            extract_one(z, relative, root / 'external_evaluation' / relative[len(start):])
    script = root / 'final_numerical_qa.py'
    extract_one(z, '04_CODE/final_numerical_qa.py', script)
    checker = module_from(script, 'archived_utility_checker')
    with contextlib.redirect_stdout(io.StringIO()):
        checker.main()
    result = json.loads((root / 'FINAL_NUMERICAL_QA.json').read_text())
    require(result['status'] == 'PASS_INDEPENDENT_NUMERICAL_REAGGREGATION' and not result['failures'], 'Utility reaggregation failed')
    result['status'] = 'PASS_REVIEW_UTILITY_REAGGREGATION'
    result['review_scope'] = 'Re-execution of the archived numerical checker on supplied completed utility/V arrays; no fitting, inference or original-cell scoring.'
    result['independent_human_validation_claimed'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('verify', 'selection', 'utilities', 'all'), default='verify')
    args = parser.parse_args()
    os.umask(0o077)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    verification = verify(args.archive)
    records = {'archive_verification': verification, 'wrapper_sha256': file_sha(Path(__file__)), 'mode': args.mode, 'new_training': False}
    with tempfile.TemporaryDirectory(prefix='scgen-review-', dir=args.out_dir) as temporary, zipfile.ZipFile(args.archive) as z:
        tmp = Path(temporary)
        if args.mode in ('selection', 'all'):
            records['selection_review'] = selection_check(z, tmp / 'selection')
        if args.mode in ('utilities', 'all'):
            records['utility_review'] = utility_check(z, tmp / 'utilities')
    records['status'] = 'PASS_CONFIDENTIAL_COMPANION_REVIEW'
    output = args.out_dir / 'REVIEW_RECEIPT.json'
    output.write_text(json.dumps(records, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': records['status'], 'receipt': str(output), 'receipt_sha256': file_sha(output), 'archive_sha256': verification['archive_sha256']}))


if __name__ == '__main__':
    main()

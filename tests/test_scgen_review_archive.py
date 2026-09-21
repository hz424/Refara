"""Recorded repackings must preserve both archive and payload verification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'analysis/scgen_training/review_companion.py'
SPEC = importlib.util.spec_from_file_location('review_companion_for_test', SOURCE)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture_archive(tmp_path, monkeypatch, *, bad_payload=False, alternate=True):
    original = b'fixed original scientific manifest\n'
    monkeypatch.setattr(review, 'MANIFEST_SHA', sha(original))
    payload = {'ORIGINAL_MANIFEST.sha256': original, 'folds/values.tsv': b'a\tb\n1\t2\n'}
    manifest = ''.join(sha(data) + '  ' + name + '\n' for name, data in payload.items()).encode()
    if bad_payload:
        payload['folds/values.tsv'] = b'a\tb\n1\t9\n'
    payload['MANIFEST.sha256'] = manifest
    archive = tmp_path / 'review.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, data in payload.items():
            z.writestr(review.PREFIX + name, data)
    profile = {'archive_sha256': sha(archive.read_bytes()), 'archive_bytes': archive.stat().st_size,
               'members': len(payload), 'manifest_sha256': sha(manifest)}
    if alternate:
        profile = {**profile, 'archive_sha256': '0' * 64, 'alternate_distributions': [profile]}
    profile_path = tmp_path / 'profiles.json'
    profile_path.write_text(json.dumps(profile))
    return archive, profile_path


@pytest.mark.parametrize('alternate', [False, True])
def test_recorded_distribution_validates_all_members(tmp_path, monkeypatch, alternate):
    archive, profile = fixture_archive(tmp_path, monkeypatch, alternate=alternate)
    result = review.verify_review_archive(archive, profile)
    assert result['archive_sha256'] == sha(archive.read_bytes())
    assert result['payload_hashes_and_crc_checked'] == 2
    assert result['model_objects_deserialized'] is False


def test_unrecorded_repacking_is_rejected(tmp_path, monkeypatch):
    archive, profile = fixture_archive(tmp_path, monkeypatch)
    raw = bytearray(archive.read_bytes())
    raw[10] ^= 1  # ZIP timestamp change leaves the payload intact.
    archive.write_bytes(raw)
    with pytest.raises(ValueError, match='Archive SHA-256 differs'):
        review.verify_review_archive(archive, profile)


def test_recorded_archive_hash_does_not_override_payload_hash(tmp_path, monkeypatch):
    archive, profile = fixture_archive(tmp_path, monkeypatch, bad_payload=True)
    with pytest.raises(ValueError, match='Archive member hash differs: folds/values.tsv'):
        review.verify_review_archive(archive, profile)


@pytest.mark.parametrize('field,value', [('status', 'SEALED'),
                                         ('scientific_design_changed', True)])
def test_protocol_projection_requires_its_recorded_scope(tmp_path, field, value):
    protocol = b'{"description":"review copy"}\n'
    (tmp_path / 'PROTOCOL.json').write_bytes(protocol)
    seal = {'status': 'SEALED_REVIEWER_PROJECTION', 'original_protocol_sha256': 'a' * 64,
            'protocol_sha256': sha(protocol),
            'reviewer_projection': {'scientific_design_changed': False}}
    if field == 'status':
        seal[field] = value
    else:
        seal['reviewer_projection'][field] = value
    (tmp_path / 'PROTOCOL_SEAL.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='Unrecognized protocol projection'):
        review.projected_selection(None, tmp_path, SimpleNamespace(PROTOCOL_SHA='a' * 64))

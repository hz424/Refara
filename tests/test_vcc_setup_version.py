"""Exercise the installer's final metadata check without installing either runtime."""
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


REPOSITORY = Path(__file__).parents[1]
INSTALLER = REPOSITORY / 'scripts/setup_vcc_runtime.sh'
METADATA_CODE = INSTALLER.read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]


def runtime_fixture(tmp_path, monkeypatch, declared_version=None, installed_version=None, source='repository'):
    repository = tmp_path/'source checkout'
    repository.mkdir()
    project_text = (REPOSITORY/'pyproject.toml').read_text()
    project = tomllib.loads(project_text)['project']
    if declared_version is not None:
        project_text = project_text.replace(f'version = "{project["version"]}"', f'version = "{declared_version}"')
    (repository/'pyproject.toml').write_text(project_text)
    expected_version = tomllib.loads(project_text)['project']['version']
    for name in ('requirements/vcc-build-py310.txt', 'requirements/vcc-build-py312.txt',
                 'requirements/vcc-py310-linux.txt', 'requirements/vcc-py312-linux.txt',
                 'requirements/unified-prepare-py310.txt', 'scripts/setup_vcc_runtime.sh'):
        target = repository/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY/name).read_bytes())
    package = repository/'src/reference_design/__init__.py'
    package.parent.mkdir(parents=True)
    package.write_text('')
    output = tmp_path/'runtime'
    output.mkdir()
    for name in ('environment_py310.txt', 'environment_official.txt'):
        (output/name).write_text('fixture==1.0\n')

    def adapter_version(name):
        assert name == project['name']
        return installed_version if installed_version is not None else expected_version

    def adapter_spec(name):
        assert name == 'reference_design'
        if source == 'missing':
            return None
        origin = package if source == 'repository' else tmp_path/'another_checkout/src/reference_design/__init__.py'
        return SimpleNamespace(origin=str(origin))

    def official_metadata(command, *, text):
        assert command[:3] == [str(output/'official/bin/python'), '-I', '-c']
        assert text is True
        return json.dumps({'python': '3.12.10', 'cell_eval2': '0.16.0'})

    monkeypatch.setattr(importlib.metadata, 'version', adapter_version)
    monkeypatch.setattr(importlib.util, 'find_spec', adapter_spec)
    monkeypatch.setattr(subprocess, 'check_output', official_metadata)
    monkeypatch.setattr(sys, 'argv', ['-', str(output), str(repository), 'fixture_commit'])
    return output, repository, expected_version


@pytest.mark.parametrize('declared_version', [None, '9.4.0.dev3'])
def test_runtime_receipt_uses_repository_version(tmp_path, monkeypatch, declared_version):
    output, repository, expected = runtime_fixture(tmp_path, monkeypatch, declared_version=declared_version)
    exec(compile(METADATA_CODE, str(INSTALLER), 'exec'), {})
    record = json.loads((output/'SETUP_COMPLETE.json').read_text())
    assert record['status'] == 'COMPLETE_RUNTIME_SETUP'
    assert record['adapter']['version'] == expected
    assert record['adapter']['editable_source'] == str(repository)
    assert record['adapter_pyproject_sha256'] == hashlib.sha256((repository/'pyproject.toml').read_bytes()).hexdigest()
    assert record['official']['cell_eval2'] == '0.16.0'
    assert not (output/'SETUP_COMPLETE.json.tmp').exists()


@pytest.mark.parametrize('failure', ['stale_version', 'foreign_source', 'missing_source'])
def test_runtime_receipt_rejects_wrong_adapter(tmp_path, monkeypatch, failure):
    options = {'installed_version': '0.0.0'} if failure == 'stale_version' else {
        'source': 'missing' if failure == 'missing_source' else 'foreign'}
    output, _, expected = runtime_fixture(tmp_path, monkeypatch, **options)
    message = f'Expected adapter {expected}; found 0.0.0' if failure == 'stale_version' else 'Expected adapter source'
    with pytest.raises(RuntimeError) as error:
        exec(compile(METADATA_CODE, str(INSTALLER), 'exec'), {})
    assert message in str(error.value)
    assert not (output/'SETUP_COMPLETE.json').exists()
    assert not (output/'SETUP_COMPLETE.json.tmp').exists()

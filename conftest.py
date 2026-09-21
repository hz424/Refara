"""Pytest policy and opt-in external Source Data fixtures."""

import os
from pathlib import Path
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY / "src"))

from perturb_nuisance_contracts.source_data_root import (  # noqa: E402
    hydrate_source_data,
    validate_source_data_root,
)


sys.dont_write_bytecode = True


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--source-data-root",
        action="store",
        default=os.environ.get("REFERENCE_CELL_SOURCE_DATA_ROOT"),
        help="optional extracted 05_SOURCE_DATA root for integration tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    candidate = config.getoption("--source-data-root")
    if candidate:
        os.environ["REFERENCE_CELL_SOURCE_DATA_ROOT"] = str(
            Path(candidate).expanduser().resolve()
        )


@pytest.fixture(scope="session")
def source_data_snapshot(pytestconfig: pytest.Config):
    candidate = pytestconfig.getoption("--source-data-root")
    if not candidate:
        pytest.skip("external Source Data root was not supplied")
    return validate_source_data_root(Path(candidate))


@pytest.fixture(scope="session")
def source_data_root(source_data_snapshot):
    return source_data_snapshot.root


@pytest.fixture(scope="session")
def hydrated_data_repository(source_data_snapshot, tmp_path_factory: pytest.TempPathFactory):
    destination = tmp_path_factory.mktemp("hydrated-source-data") / "repository"
    destination.mkdir()
    hydrate_source_data(source_data_snapshot, destination)
    return destination

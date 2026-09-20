"""Optional scGen retraining for the GSE162632 focal comparison."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import site
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .io import CapsuleError, load_array, load_manifest, read_json, read_tsv


EXPECTED_REALIZATIONS = (
    ("q1", 17),
    ("q2", 29),
    ("q3", 43),
    ("q4", 763929762),
    ("q5", 85508424),
)
EXPECTED_HYPERPARAMETERS: dict[str, Any] = {
    "n_hidden": 512,
    "n_latent": 100,
    "n_layers": 2,
    "dropout_rate": 0.2,
    "batch_size": 512,
    "terminal_epoch": 80,
    "train_size": 1.0,
    "validation_size": None,
    "early_stopping": False,
    "deterministic": True,
    "benchmark": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "logger": False,
}
RUNTIME_VERSIONS = {
    "python": "3.10.17",
    "anndata": "0.9.2",
    "scgen": "2.1.1",
    "scvi-tools": "0.20.3",
    "torch": "2.0.0+cu117",
}
SCGEN_IMPLEMENTATION_SHA256 = (
    "00348b640b5ea36355c4cdf335a89f74621e9d4d231b2d855c1190905441a527"
)
TRAIN_CELL_COLUMNS = ("row_index", "root_id", "task_id", "arm", "cell_id")
TRAIN_ROOT_TASK_COLUMNS = ("root_task_index", "root_id", "task_id")
TRAINING_ROOTS = 40
TASKS = 5
FEATURES = 2_000
CELLS_PER_ROOT_TASK_ARM = 8
TRAINING_CELLS = 3_200


@dataclass(frozen=True)
class TrainingConfiguration:
    release_version: str
    realizations: tuple[tuple[str, int], ...]
    hyperparameters: Mapping[str, Any]

    @property
    def realization_ids(self) -> tuple[str, ...]:
        return tuple(name for name, _seed in self.realizations)

    def seed_for(self, realization_id: str) -> int:
        matches = [seed for name, seed in self.realizations if name == realization_id]
        if len(matches) != 1:
            raise CapsuleError(f"Unknown training realization: {realization_id}")
        return matches[0]


def load_training_config(path: Path) -> TrainingConfiguration:
    """Read and validate the settings shared by replay and retraining."""

    document = read_json(path, "training configuration")
    if set(document) != {
        "schema_version",
        "release_version",
        "hyperparameters",
        "realizations",
    }:
        raise CapsuleError("The training configuration has unexpected fields")
    if document["schema_version"] != 1 or document["release_version"] != "1.6.1":
        raise CapsuleError("The training configuration has an unsupported version")

    hyperparameters = document["hyperparameters"]
    if not isinstance(hyperparameters, dict) or set(hyperparameters) != set(
        EXPECTED_HYPERPARAMETERS
    ):
        raise CapsuleError("The training configuration has incomplete model settings")
    for name, expected in EXPECTED_HYPERPARAMETERS.items():
        observed = hyperparameters[name]
        if type(observed) is not type(expected) or observed != expected:
            raise CapsuleError(
                f"Training setting {name!r} must be {expected!r}; observed {observed!r}"
            )

    rows = document["realizations"]
    if not isinstance(rows, list):
        raise CapsuleError("The training realization list is missing")
    realizations: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "seed"}:
            raise CapsuleError("A training realization has unexpected fields")
        name, seed = row["id"], row["seed"]
        if (
            not isinstance(name, str)
            or not name
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed < 2**32
        ):
            raise CapsuleError("A training realization ID or seed is invalid")
        realizations.append((name, seed))
    if tuple(realizations) != EXPECTED_REALIZATIONS:
        raise CapsuleError("The training realizations differ from the reported analysis")
    return TrainingConfiguration(
        release_version=document["release_version"],
        realizations=tuple(realizations),
        hyperparameters=dict(hyperparameters),
    )


def validate_manifest_realizations(
    manifest: Mapping[str, Any], config: TrainingConfiguration
) -> None:
    rows = manifest.get("realizations")
    if not isinstance(rows, list):
        raise CapsuleError("The asset manifest does not list the training realizations")
    observed: list[tuple[object, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise CapsuleError("The asset manifest contains an invalid realization")
        observed.append((row.get("id"), row.get("seed")))
    if tuple(observed) != config.realizations:
        raise CapsuleError("The asset manifest and training configuration disagree")


@dataclass(frozen=True)
class PreparedTrainingData:
    root: Path
    manifest: Mapping[str, Any]
    matrix: np.ndarray
    cells: tuple[Mapping[str, str], ...]
    root_tasks: tuple[Mapping[str, str], ...]
    feature_ids: tuple[str, ...]
    scales: np.ndarray
    weights: np.ndarray
    root_ids: tuple[str, ...]
    task_ids: tuple[str, ...]


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _integer(text: str, label: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError(f"{label} must be an integer") from error
    if str(value) != text:
        raise CapsuleError(f"{label} must use canonical integer notation")
    return value


def _finite_float(text: str, label: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError, OverflowError) as error:
        raise CapsuleError(f"{label} must be numeric") from error
    if not np.isfinite(value):
        raise CapsuleError(f"{label} must be finite")
    return value


def load_prepared_training(root: Path) -> PreparedTrainingData:
    root = Path(root)
    manifest = load_manifest(root, "gse162632_scgen_prepared_training")
    cells = tuple(read_tsv(root, manifest, "train_cells", TRAIN_CELL_COLUMNS))
    root_tasks = tuple(
        read_tsv(root, manifest, "train_root_tasks", TRAIN_ROOT_TASK_COLUMNS)
    )
    matrix = load_array(root, manifest, "train_matrix")
    feature_rows = read_tsv(
        root,
        manifest,
        "panel",
        ("feature_index", "feature_id", "scale", "weight"),
    )
    features = tuple(row["feature_id"] for row in feature_rows)
    scales = np.asarray(
        [_finite_float(row["scale"], "panel scale") for row in feature_rows],
        dtype="<f8",
    )
    weights = np.asarray(
        [_finite_float(row["weight"], "panel weight") for row in feature_rows],
        dtype="<f8",
    )
    roots = _ordered_unique([row["root_id"] for row in root_tasks])
    tasks = _ordered_unique([row["task_id"] for row in root_tasks])
    expected_pairs = tuple((root_id, task_id) for root_id in roots for task_id in tasks)
    if (
        matrix.shape != (TRAINING_CELLS, FEATURES)
        or len(cells) != TRAINING_CELLS
        or len(feature_rows) != FEATURES
        or len(root_tasks) != TRAINING_ROOTS * TASKS
        or len(roots) != TRAINING_ROOTS
        or len(tasks) != TASKS
        or roots != tuple(f"R{index:02d}" for index in range(1, TRAINING_ROOTS + 1))
        or tasks != tuple(f"T{index:02d}" for index in range(1, TASKS + 1))
        or matrix.dtype not in (np.dtype("<f4"), np.dtype("<f8"))
        or np.any(matrix < 0)
        or [_integer(row["row_index"], "training row index") for row in cells]
        != list(range(TRAINING_CELLS))
        or [
            _integer(row["root_task_index"], "training root--task index")
            for row in root_tasks
        ]
        != list(range(len(root_tasks)))
        or tuple((row["root_id"], row["task_id"]) for row in root_tasks)
        != expected_pairs
        or [
            _integer(row["feature_index"], "feature index") for row in feature_rows
        ]
        != list(range(FEATURES))
        or any(not value for value in (*features, *roots, *tasks))
        or len(set(features)) != len(features)
        or len({row["cell_id"] for row in cells}) != TRAINING_CELLS
        or any(not row["cell_id"] for row in cells)
        or np.any(scales < 0.1)
        or not np.all(weights == np.float64(1.0 / FEATURES))
        or not np.isclose(
            np.sum(weights, dtype=np.float64), 1.0, rtol=0.0, atol=1.0e-12
        )
    ):
        raise CapsuleError("The prepared training data do not match the 3200-cell panel")
    expected_pair_set = set(expected_pairs)
    groups: dict[tuple[str, str, str], int] = {}
    for row in cells:
        if row["arm"] not in {"NI", "IAV"}:
            raise CapsuleError("The training arm must be NI or IAV")
        if (row["root_id"], row["task_id"]) not in expected_pair_set:
            raise CapsuleError("A training cell is outside the root--task panel")
        key = (row["root_id"], row["task_id"], row["arm"])
        groups[key] = groups.get(key, 0) + 1
    expected_groups = {
        (root, task, arm)
        for root, task in expected_pairs
        for arm in ("NI", "IAV")
    }
    if set(groups) != expected_groups:
        raise CapsuleError("A training root--task arm is missing")
    if any(count != CELLS_PER_ROOT_TASK_ARM for count in groups.values()):
        raise CapsuleError("Each training root--task arm must contain eight cells")
    expected_cell_groups = [
        (root, task, arm)
        for root, task in expected_pairs
        for arm in ("NI", "IAV")
        for _cell in range(CELLS_PER_ROOT_TASK_ARM)
    ]
    observed_cell_groups = [
        (row["root_id"], row["task_id"], row["arm"]) for row in cells
    ]
    if observed_cell_groups != expected_cell_groups:
        raise CapsuleError(
            "Training cells must be ordered by root, task, arm and within-arm row"
        )
    return PreparedTrainingData(
        root=root,
        manifest=manifest,
        matrix=np.ascontiguousarray(matrix, dtype="<f4"),
        cells=cells,
        root_tasks=root_tasks,
        feature_ids=features,
        scales=np.ascontiguousarray(scales),
        weights=np.ascontiguousarray(weights),
        root_ids=roots,
        task_ids=tasks,
    )


def _module_origin(module: Any, label: str) -> Path:
    specification = getattr(module, "__spec__", None)
    origin = getattr(specification, "origin", None) or getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise CapsuleError(f"The imported {label} module has no verifiable source file")
    source = Path(origin)
    absolute_source = Path(os.path.abspath(source))
    if any(candidate.is_symlink() for candidate in (absolute_source, *absolute_source.parents)):
        raise CapsuleError(f"The imported {label} module has an unsafe source origin")
    try:
        resolved = source.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CapsuleError(
            f"The imported {label} module has no verifiable source file"
        ) from error
    if not resolved.is_file():
        raise CapsuleError(f"The imported {label} module has an unsafe source origin")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _user_site_roots() -> tuple[Path, ...]:
    try:
        values = site.getusersitepackages()
    except (AttributeError, OSError):
        return ()
    if isinstance(values, str):
        candidates = (values,)
    elif isinstance(values, (list, tuple)):
        candidates = tuple(values)
    else:
        candidates = ()
    roots: list[Path] = []
    for value in candidates:
        if not isinstance(value, str) or not value:
            continue
        try:
            roots.append(Path(value).expanduser().resolve(strict=False))
        except (OSError, RuntimeError):
            continue
    return tuple(dict.fromkeys(roots))


def _check_user_site_isolation(modules: Sequence[Any] = ()) -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1" or site.ENABLE_USER_SITE not in (
        False,
        None,
    ):
        raise CapsuleError(
            "GPU retraining requires user-site packages to be disabled before Python starts"
        )
    user_roots = _user_site_roots()
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        try:
            resolved = Path(entry or os.getcwd()).resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if any(_is_within(resolved, root) for root in user_roots):
            raise CapsuleError("A user-site package directory is present on sys.path")
    for module in modules:
        origin = _module_origin(module, getattr(module, "__name__", "dependency"))
        if any(_is_within(origin, root) for root in user_roots):
            raise CapsuleError("A GPU retraining dependency was imported from user site")


def _verify_scgen_implementation(
    scgen: Any, *, expected_sha256: str = SCGEN_IMPLEMENTATION_SHA256
) -> str:
    package_origin = _module_origin(scgen, "scGen")
    package_root = package_origin.parent
    if package_root.name != "scgen" or package_origin.name != "__init__.py":
        raise CapsuleError("The imported scGen package has an unexpected source layout")
    try:
        distribution = importlib.metadata.distribution("scgen")
        distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
        vae_module = importlib.import_module("scgen._scgenvae")
        model_module = importlib.import_module(scgen.SCGEN.__module__)
    except (AttributeError, ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise CapsuleError("The imported scGen implementation cannot be verified") from error
    if not distribution_root.is_dir() or not _is_within(package_root, distribution_root):
        raise CapsuleError("The imported scGen package does not match its installation record")
    vae_origin = _module_origin(vae_module, "scGen VAE")
    model_origin = _module_origin(model_module, "scGen model")
    try:
        expected_vae_origin = (package_root / "_scgenvae.py").resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CapsuleError("The imported scGen implementation cannot be verified") from error
    vae_class = getattr(vae_module, "SCGENVAE", None)
    if (
        vae_origin != expected_vae_origin
        or not _is_within(model_origin, package_root)
        or getattr(scgen.SCGEN, "__module__", "").split(".", 1)[0] != "scgen"
        or getattr(vae_class, "__module__", None) != "scgen._scgenvae"
    ):
        raise CapsuleError("The imported scGen implementation has an unsafe source origin")
    try:
        digest = hashlib.sha256(vae_origin.read_bytes()).hexdigest()
    except OSError as error:
        raise CapsuleError("The imported scGen implementation cannot be verified") from error
    if digest != expected_sha256:
        raise CapsuleError("The imported scGen implementation differs from the qualified patch")
    return digest


def check_runtime() -> dict[str, str]:
    _check_user_site_isolation()
    try:
        import anndata
        import scgen
        import scvi
        import torch
    except ImportError as error:
        raise CapsuleError(
            "GPU retraining dependencies are missing; use the supplied container or lock file"
        ) from error
    _check_user_site_isolation((np, anndata, scgen, scvi, torch))
    scgen_implementation_sha256 = _verify_scgen_implementation(scgen)
    try:
        scgen_version = importlib.metadata.version("scgen")
    except importlib.metadata.PackageNotFoundError:
        scgen_version = str(getattr(scgen, "__version__", ""))
    observed = {
        "python": platform.python_version(),
        "anndata": importlib.metadata.version("anndata"),
        "scgen": scgen_version,
        "scvi-tools": importlib.metadata.version("scvi-tools"),
        "torch": torch.__version__,
    }
    if observed != RUNTIME_VERSIONS:
        detail = ", ".join(f"{key}={value}" for key, value in observed.items())
        raise CapsuleError(f"The retraining runtime is not the qualified environment ({detail})")
    if not torch.cuda.is_available():
        raise CapsuleError("GPU retraining requires one visible CUDA GPU")
    return {
        **observed,
        "scgen_implementation_sha256": scgen_implementation_sha256,
        "user_site_packages": "disabled",
    }


def set_reproducible_seed(seed: int) -> None:
    import scvi
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise CapsuleError("The training seed must be an integer between 0 and 2**32 - 1")
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in {None, ":4096:8"}:
        raise CapsuleError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or equal to ':4096:8'"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_training_anndata(data: PreparedTrainingData) -> Any:
    import anndata
    import pandas as pd

    observations = pd.DataFrame(
        {
            "arm": pd.Categorical(
                [row["arm"] for row in data.cells],
                categories=["NI", "IAV"],
                ordered=True,
            ),
            "task": pd.Categorical(
                [row["task_id"] for row in data.cells],
                categories=list(data.task_ids),
                ordered=True,
            ),
            "root": pd.Categorical(
                [row["root_id"] for row in data.cells],
                categories=list(data.root_ids),
                ordered=True,
            ),
        },
        index=pd.Index([row["cell_id"] for row in data.cells], name="cell_id"),
    )
    variables = pd.DataFrame(index=pd.Index(data.feature_ids, name="feature_id"))
    return anndata.AnnData(X=data.matrix, obs=observations, var=variables)


def train_model(adata: Any, seed: int, config: TrainingConfiguration) -> Any:
    import scgen

    set_reproducible_seed(seed)
    hyperparameters = config.hyperparameters
    scgen.SCGEN.setup_anndata(adata, batch_key="arm", labels_key="task")
    model = scgen.SCGEN(
        adata,
        n_hidden=hyperparameters["n_hidden"],
        n_latent=hyperparameters["n_latent"],
        n_layers=hyperparameters["n_layers"],
        dropout_rate=hyperparameters["dropout_rate"],
    )
    model.train(
        max_epochs=hyperparameters["terminal_epoch"],
        use_gpu=True,
        train_size=hyperparameters["train_size"],
        validation_size=hyperparameters["validation_size"],
        batch_size=hyperparameters["batch_size"],
        early_stopping=hyperparameters["early_stopping"],
        deterministic=hyperparameters["deterministic"],
        benchmark=hyperparameters["benchmark"],
        enable_checkpointing=hyperparameters["enable_checkpointing"],
        enable_progress_bar=hyperparameters["enable_progress_bar"],
        enable_model_summary=hyperparameters["enable_model_summary"],
        logger=hyperparameters["logger"],
    )
    if not model.is_trained:
        raise CapsuleError("scGen did not reach the requested terminal epoch")
    model.module.eval()
    return model


def latent_representation(
    model: Any, adata: Any, config: TrainingConfiguration
) -> np.ndarray:
    hyperparameters = config.hyperparameters
    values = np.asarray(
        model.get_latent_representation(
            adata=adata,
            give_mean=True,
            batch_size=hyperparameters["batch_size"],
        ),
        dtype="<f4",
    )
    if (
        values.shape != (adata.n_obs, hyperparameters["n_latent"])
        or not np.isfinite(values).all()
    ):
        raise CapsuleError("scGen returned an invalid latent representation")
    return np.ascontiguousarray(values)


def _fixed_mean(values: np.ndarray, offsets: Sequence[int]) -> np.ndarray:
    if not offsets:
        raise CapsuleError("Cannot average an empty cell group")
    result = np.zeros(values.shape[1], dtype=np.float64)
    for offset in offsets:
        result += np.asarray(values[offset], dtype=np.float64)
    return result / len(offsets)


def compute_task_shifts(
    latent: np.ndarray,
    data: PreparedTrainingData,
    config: TrainingConfiguration,
) -> np.ndarray:
    if (
        latent.shape != (len(data.cells), config.hyperparameters["n_latent"])
        or not np.isfinite(latent).all()
    ):
        raise CapsuleError("The training latent matrix has the wrong shape or values")
    groups: dict[tuple[str, str, str], list[int]] = {}
    for offset, row in enumerate(data.cells):
        groups.setdefault((row["root_id"], row["task_id"], row["arm"]), []).append(
            offset
        )
    shifts = np.empty((len(data.task_ids), latent.shape[1]), dtype="<f8")
    for task_index, task_id in enumerate(data.task_ids):
        root_shifts = np.empty((len(data.root_ids), latent.shape[1]), dtype=np.float64)
        for root_index, root_id in enumerate(data.root_ids):
            root_shifts[root_index] = _fixed_mean(
                latent, groups[(root_id, task_id, "IAV")]
            ) - _fixed_mean(latent, groups[(root_id, task_id, "NI")])
        shifts[task_index] = _fixed_mean(root_shifts, range(len(data.root_ids)))
    return np.ascontiguousarray(shifts)


def module_state_sha256(model: Any) -> str:
    import torch

    digest = hashlib.sha256(b"GSE162632_SCGEN_MODULE_STATE_V1\0")
    state = model.module.state_dict()
    for key in sorted(state, key=lambda value: value.encode("utf-8")):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise CapsuleError("The scGen state contains a non-tensor value")
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

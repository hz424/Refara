from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "outputs", ".venv"}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sbatch",
    ".sh",
    ".sha256",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN = {
    "cluster data path": re.compile(r"/(?:dpc|home)/"),
    "internal account": re.compile(r"\bkuin\d+\b", re.IGNORECASE),
    "private mount": re.compile("covid" + "-m42", re.IGNORECASE),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "study-specific root label": re.compile(r"\bHMN\d+\b"),
    "production selection metadata": re.compile(
        "|".join(
            (
                "cell_" + "selection_salt",
                "evaluation_iav_" + "selection_salt",
                "control_membership_" + "salt",
            )
        )
    ),
}
FORBIDDEN_DATA_SUFFIXES = {
    ".h5",
    ".h5ad",
    ".loom",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
}
APPROVED_FOCAL_ARRAYS = {
    "capsules/gse162632_scgen/replay/arrays/c_obs.npy",
    "capsules/gse162632_scgen/replay/arrays/c_pred.npy",
    "capsules/gse162632_scgen/replay/arrays/pca64_effect.npy",
    "capsules/gse162632_scgen/replay/arrays/scales.npy",
    "capsules/gse162632_scgen/replay/arrays/scgen_native_q1.npy",
    "capsules/gse162632_scgen/replay/arrays/scgen_native_q2.npy",
    "capsules/gse162632_scgen/replay/arrays/scgen_native_q3.npy",
    "capsules/gse162632_scgen/replay/arrays/scgen_native_q4.npy",
    "capsules/gse162632_scgen/replay/arrays/scgen_native_q5.npy",
    "capsules/gse162632_scgen/replay/arrays/treated_means.npy",
    "capsules/gse162632_scgen/replay/arrays/weights.npy",
}
APPROVED_ALLOCATION_ARRAYS = {
    "capsules/gse162632_allocation/data/GSE_STOCHASTIC_UNIT_UTILITIES_V1.npy",
    "capsules/gse162632_allocation/data/GSE162632_PATTERN_UNIT_UTILITIES_V1.npy",
    "capsules/gse162632_allocation/data/GSE162632_ALLOCATION_INTERACTIONS_V1.npy",
}
APPROVED_NORMAN_ARRAYS = {"analysis/norman_reference/data/utilities.npz"}
APPROVED_NORMAN_EFFECT_ARRAYS = {"analysis/norman_direct_effect/data/effect_utilities.npz"}
APPROVED_NORMAN_TRAINING_ARRAYS = {'analysis/norman_training_review/source_data/reference_results/control_block_means.npy', 'analysis/norman_training_review/source_data/common_target_effects.npz', 'analysis/norman_training_review/source_data/direct_effect_results/effect_utilities.npz', 'analysis/norman_training_review/model/scales.npy', 'analysis/norman_training_review/source_data/reference_results/utilities.npz'}
APPROVED_MODEL_ARTIFACTS = {'analysis/norman_training_review/model/checkpoint_0400.pt'}
APPROVED_SYNTHETIC_ANNDATA = {
    f"docs/usability/materials/{task}/{name}.h5ad"
    for task in ("A", "B", "C")
    for name in ("cells", "first", "second")
}
APPROVED_TABLE1_ARRAYS = {
    "evidence/table1/empirical/task_scores.npz",
    "evidence/table1/reader_demo/profiles.npz",
    "evidence/table1/action_results/components.npz",
}
APPROVED_REPORTING_ARRAYS = {
    f"evidence/reporting_reliability/source_data/{partition}_scores.npz"
    for partition in ("C", "P", "A")
}
APPROVED_REAL_MODEL_ARRAYS = {
    f"examples/real_model_reference/data/{name}"
    for name in ("model.npz", "counts.npz", "feature_axis.npy", "scales.npy")
}
APPROVED_JERBER_AUDIT_ARRAYS = {'evidence/jerber_reference_audit/losses.npz'}
APPROVED_CONDITIONED_MODEL_ARRAYS = {
    "examples/conditioned_model_reference/data/scgen_seed17.npz",
}
APPROVED_BUNDLED_ARRAYS = (APPROVED_TABLE1_ARRAYS | APPROVED_NORMAN_TRAINING_ARRAYS | APPROVED_FOCAL_ARRAYS | APPROVED_ALLOCATION_ARRAYS
                         | APPROVED_NORMAN_ARRAYS | APPROVED_NORMAN_EFFECT_ARRAYS
                         | APPROVED_REPORTING_ARRAYS | APPROVED_REAL_MODEL_ARRAYS
                         | APPROVED_CONDITIONED_MODEL_ARRAYS | APPROVED_JERBER_AUDIT_ARRAYS)
PARSE_DERIVED_TABLES = {
    "comparator_admission/F4A_PROTOCOL_CONCORDANCE_SOURCE_V1.tsv",
    "figure3/F3_DERIVED_LEADER_SUMMARY_V1.tsv",
    "supplementary_frozen_source/parse_depth_sensitivity.tsv",
    "supplementary_frozen_source/parse_reconstructed_root_utilities.tsv",
    "v8/f3/ALL_SHARED_BASELINE_V1.tsv",
    "v8/f3/F3A_SOURCE_ROWS_V1.tsv",
    "v8/f3/F3B_SOURCE_ROWS_V1.tsv",
    "v8/f3/F3C_SOURCE_ROWS_V1.tsv",
    "v8/f4/F4_RESOURCE_METHOD_SUMMARY_V1.tsv",
    "v8/f4/F4_ROOT_METHOD_SUMMARY_V1.tsv",
}


def repository_files() -> list[Path]:
    return [
        path
        for path in REPO.rglob("*")
        if path.is_file() and not SKIP_PARTS.intersection(path.parts)
    ]


def test_no_private_paths_or_credentials() -> None:
    failures: list[str] = []
    for path in repository_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(REPO)}")
    assert not failures, "\n".join(failures)


def test_no_unapproved_raw_expression_or_model_artifacts() -> None:
    forbidden = [
        path.relative_to(REPO).as_posix()
        for path in repository_files()
        if path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES
        and path.relative_to(REPO).as_posix() not in (
            APPROVED_BUNDLED_ARRAYS | APPROVED_MODEL_ARTIFACTS | APPROVED_SYNTHETIC_ANNDATA
        )
    ]
    assert not forbidden, "raw/restricted artifacts found:\n" + "\n".join(forbidden)

    selected = REPO / "analysis/norman_training_review"
    selected_record = json.loads((selected / "model/provenance.json").read_text())
    assert hashlib.sha256((selected / "model/checkpoint_0400.pt").read_bytes()).hexdigest() == selected_record["checkpoint_sha256"]

    observed_arrays = {
        path.relative_to(REPO).as_posix()
        for path in repository_files()
        if path.suffix.lower() in {".npy", ".npz"}
    }
    assert observed_arrays == APPROVED_BUNDLED_ARRAYS

    observed_anndata = {
        path.relative_to(REPO).as_posix()
        for path in repository_files()
        if path.suffix.lower() == ".h5ad"
    }
    assert observed_anndata == APPROVED_SYNTHETIC_ANNDATA
    assert all((REPO / name).stat().st_size < 100_000 for name in observed_anndata)

    import numpy as np
    norman = REPO / "analysis/norman_reference/data"
    report = json.loads((norman / "score_report.json").read_text())
    assert hashlib.sha256((norman / "utilities.npz").read_bytes()).hexdigest() == report["output_sha256"]["utilities.npz"]
    with np.load(norman / "utilities.npz", allow_pickle=False) as arrays:
        assert {name: arrays[name].shape for name in arrays.files} == {
            "utility": (55, 30, 6, 4, 5), "atomic_utility": (55, 30, 6, 4, 27),
            "V": (30, 6), "K": (55, 30, 6, 3),
            "conditioning_state_distance_squared": (55, 30, 6, 3),
            "tasks": (55,), "models": (4,), "patterns": (5,), "depths": (6,),
            "role_tuples": (27, 3),
        }
    effect_root = REPO / "analysis/norman_direct_effect"
    effect_report = json.loads((effect_root / "results/score_receipt.json").read_text())
    effect_path = effect_root / "data/effect_utilities.npz"
    assert hashlib.sha256(effect_path.read_bytes()).hexdigest() == effect_report["output_sha256"]["effect_utilities.npz"]
    assert hashlib.sha256((norman / "utilities.npz").read_bytes()).hexdigest() == effect_report["original_utilities_sha256"]
    with np.load(effect_path, allow_pickle=False) as arrays:
        assert {name: arrays[name].shape for name in arrays.files} == {
            "conditions": (55,), "patterns": (5,), "state_models": (3,),
            "depths": (6,), "role_tuples": (27, 3),
            "effect_utility": (55, 30, 6, 5), "effect_atomic_utility": (55, 30, 6, 27),
            "V": (30, 6), "d_S": (55, 30, 6, 3), "d_D": (55, 30, 6, 3),
        }
        for name in ["effect_utility", "effect_atomic_utility", "V", "d_S", "d_D"]:
            assert np.isfinite(arrays[name]).all(), name
        assert np.max(np.ptp(arrays["effect_utility"], axis=-1)) <= 1e-12
        np.testing.assert_allclose(arrays["d_D"], arrays["d_S"] + arrays["V"][..., None], rtol=0, atol=1e-12)


def test_reader_facing_title_is_current() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    assert (
        "Reference-cell design shapes model evaluation in single-cell perturbation prediction"
    ) in normalized
    assert (
        "Single-cell perturbation benchmark rankings can reverse across "
        "reference-cell constructions"
    ) not in normalized


def test_software_citation_metadata_is_complete_without_deposit_claims() -> None:
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    package = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    package_version = re.search(r'^version = "([^"]+)"$', package, re.MULTILINE).group(1)
    assert f'version: "{package_version}"' in citation
    assert "College of Medicine and Health Sciences, Department of Biological Sciences" in citation
    assert "Center for Biotechnology (BTC)" in citation
    assert "hao.zhou@ku.ac.ae" in citation
    assert 'date-released: "2026-09-20"' in citation
    assert "doi:" not in citation.lower()
    assert "orcid:" not in citation.lower()


def test_parse_license_boundary_is_explicit_and_complete() -> None:
    scope = (REPO / "data/derived/LICENSE_SCOPE.md").read_text(encoding="utf-8")
    for relative in PARSE_DERIVED_TABLES:
        assert not (REPO / "data/derived" / relative).exists(), relative
        assert f"`{relative}`" in scope, relative

    assert "CC BY-NC 4.0" in scope
    assert (REPO / "LICENSES/CC-BY-NC-4.0.md").is_file()
    assert not (REPO / "figures/supporting/requirements.txt").exists()


def test_table1_approved_arrays_are_exact_manifest_bound_derived_evidence() -> None:
    """Whitelist only three saved-score/profile outputs, with explicit shape/content checks."""
    import numpy as np
    bundle = REPO / 'evidence/table1'
    manifest = json.loads((bundle / 'BUNDLE_MANIFEST.json').read_text())
    indexed = {r['path']: r for r in manifest['files']}
    for name in APPROVED_TABLE1_ARRAYS:
        path = REPO / name
        relative = path.relative_to(bundle).as_posix()
        rec = indexed[relative]
        assert path.stat().st_size == rec['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256']

    with np.load(bundle / 'empirical/task_scores.npz', allow_pickle=False) as arrays:
        assert arrays['scores_direct_state'].shape == (55, 6, 30, 3, 6, 4)
        assert arrays['scores_role_effect'].shape == (55, 6, 30, 27, 6, 4)
        assert arrays['depths'].tolist() == [8, 16, 32, 64, 128, 135]
        assert arrays['tasks'].shape == (55,)
        assert arrays['role_tuples'].shape == (27, 3)
        assert np.isfinite(arrays['scores_direct_state']).all()
        assert np.isfinite(arrays['scores_role_effect']).all()

    profiles = json.loads((bundle / 'reader_demo/manifest.json').read_text())['array_manifest']
    with np.load(bundle / 'reader_demo/profiles.npz', allow_pickle=False) as arrays:
        assert set(arrays.files) == set(profiles)
        for name in arrays.files:
            a = np.ascontiguousarray(arrays[name], dtype='<f8')
            assert list(a.shape) == profiles[name]['shape']
            digest = hashlib.sha256(json.dumps(list(a.shape), separators=(',', ':')).encode() + a.tobytes()).hexdigest()
            assert digest == profiles[name]['sha256']
            assert np.isfinite(a).all()

    cases = json.loads((bundle / 'action_results/cases.json').read_text())
    case_ids = {case['case_id'] for case in cases}
    assert len(cases) == len(case_ids) == 95
    with np.load(bundle / 'action_results/components.npz', allow_pickle=False) as arrays:
        assert {name.split('__')[0] for name in arrays.files} == case_ids
        assert {name.split('__')[1] for name in arrays.files} == {'before', 'after'}
        for name in arrays.files:
            value = arrays[name]
            assert value.ndim in (1, 2)
            assert np.issubdtype(value.dtype, np.number)
            assert not np.isinf(value).any()  # Defined metric NA values remain NaN.

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from itertools import combinations
from pathlib import Path


HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE.parent
SOURCE = ADAPTER_DIR / "sound_life_matching_robustness_v2.py"
SPEC = ADAPTER_DIR / "spec.json"

module_spec = importlib.util.spec_from_file_location("sound_life_v2", SOURCE)
assert module_spec is not None and module_spec.loader is not None
adapter = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = adapter
module_spec.loader.exec_module(adapter)


def external_inputs_available() -> bool:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    root = (SPEC.parent / spec["project_root_relative_to_spec"]).resolve()
    return root.is_dir() and all(
        (root / item["path"]).is_file() for item in spec["inputs"]
    )


def tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class ExactOptimizationTests(unittest.TestCase):
    def test_implementation_receipt_uses_repository_relative_paths(self) -> None:
        receipt = adapter.implementation_receipt()
        repository_root = SOURCE.parents[2]
        expected = {
            "source_path": (
                "analysis/sound_life_matching/"
                "sound_life_matching_robustness_v2.py"
            ),
            "test_path": (
                "analysis/sound_life_matching/tests/"
                "test_sound_life_matching_robustness_v2.py"
            ),
            "analysis_spec_path": (
                "analysis/sound_life_matching/"
                "SOUND_LIFE_ROBUST_MATCHING_ANALYSIS_SPEC_V2.md"
            ),
        }
        for path_key, relative in expected.items():
            self.assertEqual(receipt[path_key], relative)
            path = repository_root / relative
            self.assertTrue(path.is_file())
            hash_key = path_key.replace("_path", "_sha256")
            self.assertEqual(receipt[hash_key], adapter.sha256_file(path))

    def test_exact_min_cost_matches_brute_force(self) -> None:
        edges = (
            adapter.Edge("D1", "Y1", "B1", "P", 1, 1),
            adapter.Edge("D1", "Y2", "B2", "P", 1, 1),
            adapter.Edge("D2", "Y1", "B1", "P", 1, 1),
            adapter.Edge("D2", "Y2", "B2", "P", 1, 1),
            adapter.Edge("D3", "Y1", "B2", "P", 1, 1),
        )
        weights = (
            Fraction(1, 3),
            Fraction(-2, 7),
            Fraction(5, 11),
            Fraction(3, 5),
            Fraction(-1, 13),
        )
        feasible = []
        for candidate in combinations(edges, 2):
            if (
                len({edge.donor_id for edge in candidate}) == 2
                and len({edge.batch_id for edge in candidate}) == 2
            ):
                feasible.append(candidate)
        brute_values = [
            sum((weights[edges.index(edge)] for edge in matching), Fraction(0))
            for matching in feasible
        ]
        minimum, min_matching = adapter.optimize_additive(
            edges, 2, weights, "MIN"
        )
        maximum, max_matching = adapter.optimize_additive(
            edges, 2, weights, "MAX"
        )
        self.assertEqual(minimum, min(brute_values))
        self.assertEqual(maximum, max(brute_values))
        self.assertEqual(len(min_matching), 2)
        self.assertEqual(len(max_matching), 2)

    def test_exact_enumerator_is_complete_on_small_graph(self) -> None:
        edges = (
            adapter.Edge("D1", "Y1", "B1", "P", 1, 1),
            adapter.Edge("D2", "Y1", "B1", "P", 1, 1),
            adapter.Edge("D2", "Y2", "B2", "P", 1, 1),
            adapter.Edge("D3", "Y2", "B2", "P", 1, 1),
        )
        observed = adapter.enumerate_maximum_matchings(edges, 2, 3)
        self.assertEqual(len(observed), 3)
        self.assertEqual(
            len({adapter.membership_digest(item) for item in observed}), 3
        )


@unittest.skipUnless(
    external_inputs_available(),
    "restricted Sound Life inputs are not bundled; see docs/DATA_ACCESS.md",
)
class ActualBundleIntegrationTests(unittest.TestCase):
    def test_build_and_verify_actual_hash_bound_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            adapter.build(SPEC, bundle)
            adapter.verify(SPEC, bundle)
            result = json.loads(
                (
                    bundle
                    / "SOUND_LIFE_MATCHING_ROBUSTNESS_RESULT_V2.json"
                ).read_text(encoding="utf-8")
            )
            graphs = {row["graph_id"]: row for row in result["graphs"]}
            self.assertEqual(
                graphs[
                    "SAME_BATCH_SAME_POOL_TRAIN_BATCH_EXCLUDED"
                ]["exhaustive_maximum_matching_count"],
                144,
            )
            self.assertTrue(
                graphs[
                    "SAME_BATCH_SAME_POOL_TRAIN_BATCH_EXCLUDED"
                ]["all_maximum_matchings_exhaustively_enumerated"]
            )
            self.assertFalse(
                graphs["SAME_BATCH_SAME_POOL"][
                    "all_maximum_matchings_exhaustively_enumerated"
                ]
            )
            self.assertFalse(
                result["authority_boundary"]["confirmatory_authority"]
            )

            leaders = tsv_rows(
                bundle / "SOUND_LIFE_LEADER_ROBUSTNESS_V2.tsv"
            )
            self.assertEqual(len(leaders), 16)
            self.assertFalse(
                any(row["classification"] == "" for row in leaders)
            )
            holm_rows = tsv_rows(
                bundle / "SOUND_LIFE_HOLM_ROBUSTNESS_V2.tsv"
            )
            self.assertEqual(len(holm_rows), 112)
            exact_holm = [
                row
                for row in holm_rows
                if row["graph_id"]
                == "SAME_BATCH_SAME_POOL_TRAIN_BATCH_EXCLUDED"
            ]
            self.assertFalse(
                any("UNRESOLVED" in row["classification"] for row in exact_holm)
            )

            target = bundle / "SOUND_LIFE_MATCHING_MEMBERSHIP_BOUNDS_V2.tsv"
            target.write_bytes(target.read_bytes() + b"tamper\n")
            with self.assertRaises(adapter.RobustnessError):
                adapter.verify(SPEC, bundle)


if __name__ == "__main__":
    unittest.main()

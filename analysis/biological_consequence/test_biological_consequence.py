"""Synthetic tests only; these fixtures contain no biological measurements."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import biological_consequence as bc
import summarize_program_interpretation as spi


class ConsequenceTests(unittest.TestCase):
    def test_interpretation_threshold_and_stable_top_rules(self):
        values = np.array([[-.05, -.01, 0., .01, .05]])
        np.testing.assert_array_equal(spi.active_sign(values, .01), [[-1, 0, 0, 0, 1]])

        tied = np.array([[3., -3., 2., -2.]])
        np.testing.assert_array_equal(spi.stable_top(tied, 3), [[0, 1, 2]])
        self.assertFalse(spi.changed_top_sets(tied, [[-3., 3., 2., -2.]], 2)[0])
        self.assertTrue(spi.changed_top_sets(tied, [[-3., 2., 3., -2.]], 2)[0])

    def test_complete_post_hoc_interpretation_decomposition(self):
        programs = np.asarray([f"H{i}" for i in range(37)])
        block_pairs = np.asarray([[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]])
        observed = np.linspace(1., .1, 37)
        selected_s = observed * .9
        selected_d = selected_s.copy()
        selected_d[-1] *= -1
        scores = np.empty((10, 6, 40, 3, 37), dtype=np.float64)
        scores[..., 0, :] = observed
        scores[..., 1, :] = selected_s
        scores[..., 2, :] = selected_d

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "program_readouts.npz"
            np.savez(
                archive,
                scores=scores,
                program_ids=programs,
                score_axis=np.asarray(["observed", "S_selected", "D_selected"]),
                label_ids=np.arange(0, 1000, 100),
                block_pairs=block_pairs,
                root_index=np.repeat(np.arange(8), 5),
                task_index=np.tile(np.arange(5), 8),
            )
            table = root / "program_scores.tsv"
            with table.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_row", "root_id", "task_id", "cell_type", "program_id",
                        "observed", "S_selected", "D_selected",
                    ],
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                cell_types = ["B", "CD4_T", "CD8_T", "monocytes_combined", "NK_combined"]
                for task in range(40):
                    for program in range(37):
                        writer.writerow({
                            "task_row": task,
                            "root_id": f"R{task // 5}",
                            "task_id": f"T{task % 5}",
                            "cell_type": cell_types[task % 5],
                            "program_id": programs[program],
                            "observed": observed[program],
                            "S_selected": selected_s[program],
                            "D_selected": selected_d[program],
                        })

            output = root / "interpretation"
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(spi.__file__).resolve()),
                    "--program-scores",
                    str(table),
                    "--program-readouts",
                    str(archive),
                    "--output-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            receipt = json.loads((output / "INTERPRETATION_RECEIPT.json").read_text())
            summary = json.loads((output / "interpretation_summary.json").read_text())
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(summary["status"], "PASS_POST_HOC_DESCRIPTIVE_SUMMARY")
            self.assertEqual(
                summary["primary_direction"]["0.0"]
                ["tasks_with_at_least_one_opposite_active_direction"],
                40,
            )
            self.assertEqual(summary["S_D_same_largest_absolute_score_program_task_settings"], 1200)
            self.assertEqual(summary["largest_absolute_score_program_common_to_all_S_D_task_settings"], "H0")
            self.assertEqual(len(receipt["outputs_sha256"]), 9)

    def test_held_out_donor_never_affects_own_selection(self):
        u = np.zeros((1000, 5, 8, 8))
        u[:, (0, 4), :, 1] = 1
        original, _, records = bc.select_models(u)
        u[:, :, 0, 7] = 1e6
        changed, _, _ = bc.select_models(u)
        np.testing.assert_array_equal(original[0], changed[0])
        self.assertTrue(np.all(changed[1:] == 7))
        self.assertTrue(all(r["root_index"] not in r["selection_root_indices"] for r in records))

    def test_selection_tolerance_preserves_full_maximizer_set(self):
        u = np.zeros((3, 5, 3, 8))
        u[:, (0, 4), :, 0] = 1 - 5e-13
        u[:, (0, 4), :, 1] = 1
        winners, _, records = bc.select_models(u)
        self.assertTrue(np.all(winners == 0))
        self.assertTrue(all(r["maximizer_indices"] == [0, 1] for r in records))
        self.assertTrue(all(r["top_two_gap"] > 0 for r in records))

    def test_state_output_centred_once_and_direct_effect_unchanged(self):
        d = {"root_index": np.array([0]), "treated_means": np.zeros((1, 2)),
             "control_means": np.full((1, 3, 1, 2), 4, dtype=np.float32),
             "state_predictions": np.full((1, 3, 1, 3, 2), 10, dtype=np.float32),
             "direct_effects": np.full((1, 5, 2), 3, dtype=np.float64)}
        effects = bc.selected_effects(d, np.array([[5, 1]]), 0, 0)
        np.testing.assert_array_equal(effects[0, 0], [6, 6])
        np.testing.assert_array_equal(effects[0, 1], [3, 3])
        self.assertEqual(effects.dtype, np.dtype("float64"))
        same = bc.selected_effects(d, np.array([[6, 6]]), 0, 0)
        np.testing.assert_array_equal(same[:, 0], same[:, 1])
        z = bc.program_scores(same, [2, 4], [[0, 1]])
        self.assertEqual(float(np.abs(z[:, 0] - z[:, 1]).mean()), 0)

    def test_zero_response_and_empty_top_genes(self):
        z = bc.program_scores(np.zeros((2, 4)), [1, 2, 3, 4], [[0, 1], [2, 3]])
        np.testing.assert_array_equal(z, np.zeros((2, 2)))
        np.testing.assert_array_equal(bc.absolute_ranks(z), np.ones((2, 2)))
        self.assertEqual(len(bc.top_genes(np.zeros(4), 50)), 0)
        empty = bc.gene_overlap(np.zeros(4), np.zeros(4), 50)
        self.assertTrue(empty["both_empty"])
        self.assertEqual(empty["jaccard"], 1)
        self.assertEqual(empty["direction_consistent_overlap"], 1)
        self.assertEqual(empty["union_size"], 0)
        one = bc.gene_overlap(np.zeros(4), np.ones(4), 50)
        self.assertTrue(one["one_empty"])
        self.assertEqual(one["jaccard"], 0)

    def test_signed_overlap_and_original_gene_order_ties(self):
        a = np.array([2., -2., 0., 1.])
        b = np.array([-2., -2., 0., 1.])
        np.testing.assert_array_equal(bc.top_genes(a, 2), [0, 1])
        overlap = bc.gene_overlap(a, b, 2)
        self.assertEqual(overlap["jaccard"], 1)
        self.assertEqual(overlap["direction_consistent_overlap"], .5)

    def test_equal_donor_aggregation_not_pooled_rows(self):
        donor, overall = bc.aggregate(np.array([[0.], [2.], [10.]]), np.array([0, 0, 1]))
        np.testing.assert_array_equal(donor, [[1.], [10.]])
        np.testing.assert_array_equal(overall, [5.5])

    def test_program_shift_is_signed_scaled_mean(self):
        z = bc.program_scores([[2., -4., 12.]], [2., 2., 3.], [[0, 1], [2]])
        np.testing.assert_array_equal(z, [[-.5, 4.]])

    def test_representative_case_keeps_median_tie_order(self):
        row, top, median = bc.representative_case(np.array([1., 2., 4., 5.]),
                                                  np.array([[1., 0.], [-3., 3.], [0., 0.], [0., 0.]]))
        self.assertEqual(median, 3)
        self.assertEqual(row, 1)
        np.testing.assert_array_equal(top, [0, 1])

    def test_production_axis_contract_rejects_reordered_rows(self):
        d = {"selection_utilities": np.zeros((1000, 5, 8, 8)),
             "state_predictions": np.zeros((10, 3, 40, 3, 2000), dtype=np.float32),
             "direct_effects": np.zeros((40, 5, 2000)),
             "control_means": np.zeros((10, 3, 40, 2000), dtype=np.float32),
             "treated_means": np.zeros((40, 2000)), "scales": np.ones(2000),
             "weights": np.full(2000, 1 / 2000),
             "root_index": np.repeat(np.arange(8), 5), "task_index": np.tile(np.arange(5), 8),
             "label_ids": np.asarray(bc.LABELS), "selection_label_ids": np.arange(1000)}
        models = list(bc.MODEL_IDS)
        axes = {"model_ids": models, "direct_model_ids": models[:5], "state_model_ids": models[5:],
                "model_output_kinds": ["direct_effect"] * 5 + ["absolute_state"] * 3,
                "root_ids": [f"R{i}" for i in range(8)], "task_ids": [f"T{i}" for i in range(5)],
                "gene_ids": [f"G{i}" for i in range(2000)], "pattern_ids": ["S", "M", "P", "O", "D"]}
        mapping = {"gene_ids": axes["gene_ids"], "programs": [
            {"program_id": f"H{i}", "panel_gene_indices": list(range(15)),
             "mapped_member_count": 100, "official_unique_member_count": 100, "eligible": True}
            for i in range(50)]}
        coverage, members, _ = bc.validate_inputs(d, axes, mapping)
        self.assertEqual(len(members), 50)
        self.assertEqual(coverage[0]["coverage_fraction"], .15)
        d["task_index"][[0, 1]] = d["task_index"][[1, 0]]
        with self.assertRaisesRegex(ValueError, "original donor-major"):
            bc.validate_inputs(d, axes, mapping)

    def test_complete_synthetic_output_aggregation_and_sensitivity(self):
        n, g = 40, 4
        u = np.zeros((1000, 5, 8, 8))
        u[:, 0, :, 5] = 2
        u[:, 4, :, 3] = 3
        control = np.ones((10, 3, n, g), dtype=np.float32)
        d = {"selection_utilities": u, "root_index": np.repeat(np.arange(8), 5),
             "task_index": np.tile(np.arange(5), 8), "control_means": control,
             "state_predictions": np.full((10, 3, n, 3, g), 3, dtype=np.float32),
             "direct_effects": np.full((n, 5, g), 4.), "treated_means": np.full((n, g), 4.),
             "scales": np.array([1., 2., 4., 8.])}
        d["direct_effects"][:, 0] = 0
        axes = {"model_ids": [f"M{i}" for i in range(8)], "root_ids": [f"R{i}" for i in range(8)],
                "task_ids": [f"T{i}" for i in range(5)]}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = bc.analyze(d, axes, [[0, 1], [2, 3]], ["P0", "P1"], out)
            self.assertEqual(result["reference_assignment_count"], 60)
            self.assertEqual(result["primary_example_task_row"], 0)
            example = json.loads((out / "example_selection.json").read_text())
            self.assertEqual(example["selected_model_S"], "M5")
            self.assertEqual(example["selected_model_D"], "M3")
            with np.load(out / "program_readouts.npz", allow_pickle=False) as z:
                self.assertEqual(z["scores"].shape, (10, 6, 40, 3, 2))
                np.testing.assert_array_equal(z["scores"][0, 0, 0], [[2.25, .5625], [1.5, .375], [3., .75]])
                np.testing.assert_array_equal(z["selected_model_indices"], np.tile([5, 3], (8, 1)))
            self.assertEqual(len((out / "task_summary.tsv").read_text().splitlines()), 41)
            self.assertEqual(len((out / "reference_sensitivity.tsv").read_text().splitlines()), 2401)
            self.assertEqual(len((out / "gene_overlaps.tsv").read_text().splitlines()), 21601)


if __name__ == "__main__":
    unittest.main()

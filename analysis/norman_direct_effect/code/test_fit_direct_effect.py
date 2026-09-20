"""Synthetic fitting checks; no Norman test or reference data are loaded."""
import hashlib
import unittest

import numpy as np

from fit_direct_effect import (components, cross_validate, double_folds, fold_indices,
                               make_design, ridge_coefficients, select_alpha)


class TestNativeEffectRidge(unittest.TestCase):
    def setUp(self):
        self.features = ["A", "B", "C", "D"]
        self.conditions = ["ctrl", *self.features, "A+B", "A+C", "A+D", "B+C", "B+D", "C+D"]
        self.x = make_design(self.conditions, self.features)
        self.target = self.x @ np.array([[1.0, 2.0], [-2.0, 0.5], [0.5, 3.0], [-1.0, 1.0]])
        self.target[5:] += np.arange(12).reshape(6, 2) / 10

    def test_zero_effect_at_control_and_no_intercept(self):
        coefficients = ridge_coefficients(self.x, self.target, 0.1)
        np.testing.assert_array_equal(make_design(["ctrl"], self.features) @ coefficients, np.zeros((1, 2)))
        self.assertEqual(coefficients.shape, (4, 2))

    def test_regularized_solution_matches_augmented_least_squares(self):
        for alpha in [0, 0.01, 1, 100]:
            expected = np.linalg.lstsq(np.vstack([self.x, np.sqrt(alpha) * np.eye(4)]),
                                       np.vstack([self.target, np.zeros((4, 2))]), rcond=None)[0]
            np.testing.assert_allclose(ridge_coefficients(self.x, self.target, alpha), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(ridge_coefficients(np.eye(2), np.array([[2., 6.], [4., 8.]]), 3),
                                   np.array([[0.5, 1.5], [1., 2.]]), rtol=0, atol=0)

    def test_hash_split_is_condition_based_and_order_invariant(self):
        first = double_folds(self.conditions, "example", 3)
        self.assertEqual(first, double_folds(self.conditions[::-1], "example", 3))
        ordered = sorted(self.conditions[5:], key=lambda c: (hashlib.sha256(("example\0" + c).encode()).hexdigest(), c))
        self.assertEqual({c: first[c][0] for c in first}, {c: i % 3 for i, c in enumerate(ordered)})
        self.assertEqual([sum(v[0] == f for v in first.values()) for f in range(3)], [2, 2, 2])

    def test_validation_targets_are_excluded_from_fit(self):
        assignments = double_folds(self.conditions, "example", 3)
        all_held = []
        for fold in range(3):
            train, held = fold_indices(self.conditions, assignments, fold)
            self.assertTrue(set(range(5)).issubset(train))
            self.assertFalse(set(train) & set(held))
            all_held.extend(held)
            first = ridge_coefficients(self.x[train], self.target[train], 1)
            poisoned = self.target.copy()
            poisoned[held] += 1e9
            second = ridge_coefficients(self.x[train], poisoned[train], 1)
            np.testing.assert_array_equal(first, second)
        self.assertEqual(sorted(all_held), list(range(5, 11)))

    def test_cross_validation_has_one_prediction_per_double_per_alpha(self):
        result = cross_validate(self.x, self.target, self.conditions, [0, 1], "example", 3)
        assignments, doubles, predictions, rows, train_rows, fold_rows, alpha_rows = result
        self.assertEqual(predictions.shape, (2, 6, 2))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(train_rows), 2 * 3 * 9)
        for alpha_index, alpha in enumerate([0, 1]):
            errors = np.mean((predictions[alpha_index] - self.target[[self.conditions.index(c) for c in doubles]]) ** 2, axis=1)
            self.assertAlmostEqual(errors.mean(), alpha_rows[alpha_index]["validation_mse"], places=15)
            for condition in doubles:
                self.assertEqual(sum(r["alpha"] == alpha and r["condition"] == condition for r in rows), 1)

    def test_selection_is_deterministic_and_uses_absolute_tie_tolerance(self):
        alpha, value = select_alpha([0, 0.01, 0.1, 1, 10, 100], [2, 1 + 3e-13, 1, 1 + 8e-13, 1 + 2e-12, 2], 1e-12)
        self.assertEqual(alpha, 1)
        self.assertEqual(value, 1)
        self.assertEqual(select_alpha([0, 1, 100], [3, 2, 1], 1e-12)[0], 100)
        self.assertEqual(select_alpha([100, 1, 0], [1, 2, 3], 1e-12)[0], 100)

    def test_invalid_inputs_fail_explicitly(self):
        for condition in ["", " ctrl", "A+A", "A+ctrl", "A+B+C"]:
            with self.assertRaises(ValueError):
                components(condition)
        with self.assertRaises(ValueError):
            make_design(["A+E"], self.features)
        with self.assertRaises(ValueError):
            ridge_coefficients(self.x, self.target, -1)
        with self.assertRaises(ValueError):
            select_alpha([0, 1], [float("nan"), 1], 1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)

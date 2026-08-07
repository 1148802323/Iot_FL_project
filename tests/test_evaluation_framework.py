import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluator", ROOT / "standalone_non_iid_evaluator.py")
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


class EvaluationFrameworkTests(unittest.TestCase):
    def test_validation_threshold_is_not_fixed_at_half(self):
        y = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.05, 0.10, 0.20, 0.25])
        self.assertLess(evaluator.tune_threshold(y, probability), 0.5)

    def test_cost_metrics_penalise_false_negatives(self):
        y = np.asarray([0, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.9])
        result = evaluator.metrics(
            y, probability, 0.5, false_negative_cost=10, false_positive_cost=1
        )
        self.assertEqual(result["total_cost"], 10)
        self.assertTrue(math.isclose(result["cost_per_1000"], 10000 / 3))

    def test_invalid_probabilities_are_rejected(self):
        for probability in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                frame = pd.DataFrame({
                    "UDI": [1], "seed": [42], "strategy": ["iid"],
                    "split": ["test"], "probability": [probability],
                })
                evaluator.validate_probability_frame(frame, Path("invalid.csv"))

    def test_duplicate_final_rows_are_rejected_by_alignment(self):
        submitted = pd.DataFrame({"UDI": [1, 1], "probability": [0.2, 0.3]})
        expected = pd.DataFrame({"UDI": [1], "Machine failure": [0]})
        with self.assertRaises(ValueError):
            evaluator.aligned_probabilities(submitted, expected, "duplicate test")


if __name__ == "__main__":
    unittest.main()

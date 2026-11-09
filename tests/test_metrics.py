"""
Unit tests for recommender system evaluation metrics in src/metrics.py.
"""

import math
import pytest
from src.metrics import rmse, precision_at_k, recall_at_k, ndcg_at_k


class TestRMSE:
    def test_rmse_perfect_predictions(self):
        y_true = [4.0, 5.0, 2.0, 3.5]
        y_pred = [4.0, 5.0, 2.0, 3.5]
        assert rmse(y_true, y_pred) == 0.0

    def test_rmse_known_values(self):
        # differences: [4-3=1, 2-4=-2] -> squares: [1, 4] -> mean=2.5 -> sqrt(2.5)
        y_true = [3.0, 4.0]
        y_pred = [4.0, 2.0]
        expected = math.sqrt(2.5)
        assert pytest.approx(rmse(y_true, y_pred), abs=1e-5) == expected

    def test_rmse_empty(self):
        assert rmse([], []) == 0.0

    def test_rmse_shape_mismatch(self):
        with pytest.raises(ValueError):
            rmse([1.0, 2.0], [1.0])


class TestPrecisionAtK:
    def test_precision_all_hits(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5}
        assert precision_at_k(recommended, relevant, k=5) == 1.0

    def test_precision_partial_hits(self):
        recommended = [10, 20, 30, 40, 50]
        relevant = {20, 40, 99}
        # top 5 has 20 and 40 (2 hits / 5) = 0.4
        assert precision_at_k(recommended, relevant, k=5) == 0.4
        # top 2 has only 20 (1 hit / 2) = 0.5
        assert precision_at_k(recommended, relevant, k=2) == 0.5

    def test_precision_zero_hits(self):
        recommended = [1, 2, 3]
        relevant = {7, 8, 9}
        assert precision_at_k(recommended, relevant, k=3) == 0.0

    def test_precision_empty(self):
        assert precision_at_k([], {1, 2}, k=5) == 0.0
        assert precision_at_k([1, 2], set(), k=5) == 0.0


class TestRecallAtK:
    def test_recall_full(self):
        recommended = [10, 20, 30, 40]
        relevant = {10, 20}
        # Both relevant items found in top 4 -> 2 / 2 = 1.0
        assert recall_at_k(recommended, relevant, k=4) == 1.0

    def test_recall_partial(self):
        recommended = [10, 20, 30, 40, 50]
        relevant = {20, 40, 99}  # 3 relevant items
        # Top 5 contains 20 and 40 (2 out of 3)
        assert pytest.approx(recall_at_k(recommended, relevant, k=5), abs=1e-5) == 2 / 3
        # Top 2 contains only 20 (1 out of 3)
        assert pytest.approx(recall_at_k(recommended, relevant, k=2), abs=1e-5) == 1 / 3

    def test_recall_empty_relevant(self):
        assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


class TestNDCGAtK:
    def test_ndcg_perfect_ranking(self):
        recommended = [10, 20, 30, 40, 50]
        relevant = {10, 20}
        # Top 2 are both hits -> perfect ranking -> NDCG = 1.0
        assert ndcg_at_k(recommended, relevant, k=5) == 1.0

    def test_ndcg_known_values(self):
        # Recommended: [1, 2, 3, 4, 5]
        # Relevant: {1, 3} (2 relevant items)
        # DCG@5 = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@5 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975... = 1.63092975...
        # Expected NDCG = 1.5 / 1.63092975... = 0.91972...
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3}
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        expected = dcg / idcg
        assert pytest.approx(ndcg_at_k(recommended, relevant, k=5), abs=1e-5) == expected

    def test_ndcg_zero_hits(self):
        recommended = [1, 2, 3]
        relevant = {7, 8}
        assert ndcg_at_k(recommended, relevant, k=3) == 0.0

    def test_ndcg_empty(self):
        assert ndcg_at_k([], {1, 2}, k=5) == 0.0
        assert ndcg_at_k([1, 2], set(), k=5) == 0.0

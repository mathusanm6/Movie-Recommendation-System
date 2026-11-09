import math
from typing import Sequence, Set, Union
import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Computes root mean squared error between actual and predicted ratings.

    Args:
        y_true: Ground truth ratings.
        y_pred: Predicted ratings.

    Returns:
        RMSE value as a float. Returns 0.0 if empty.
    """
    if len(y_true) == 0:
        return 0.0

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"Shape mismatch: {y_true_arr.shape} vs {y_pred_arr.shape}")

    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def precision_at_k(
    recommended: Sequence[Union[int, str]],
    relevant: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """
    Computes Precision@K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set or list of ground truth relevant item IDs.
        k: Number of recommendations to evaluate.

    Returns:
        Fraction of recommended items in the top K that are relevant.
    """
    if k <= 0:
        return 0.0

    top_k = recommended[:k]
    if len(top_k) == 0:
        return 0.0

    relevant_set = set(relevant)
    hits = sum(1 for item in top_k if item in relevant_set)
    return float(hits / k)


def recall_at_k(
    recommended: Sequence[Union[int, str]],
    relevant: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """
    Computes Recall@K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set or list of ground truth relevant item IDs.
        k: Number of recommendations to evaluate.

    Returns:
        Fraction of total relevant items captured in the top K recommendations.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0

    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return float(hits / len(relevant_set))


def ndcg_at_k(
    recommended: Sequence[Union[int, str]],
    relevant: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set or list of ground truth relevant item IDs.
        k: Number of recommendations to evaluate.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0

    top_k = recommended[:k]

    dcg = 0.0
    for rank_idx, item in enumerate(top_k):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank_idx + 2)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_hits))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)

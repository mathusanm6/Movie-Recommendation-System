"""
Evaluation pipeline for benchmark comparison across recommender models.
Runs temporal train-test split, fits baselines, Item-KNN, and PyTorch MF,
and computes RMSE, Precision@10, Recall@10, and NDCG@10.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data import (
    load_ratings,
    temporal_train_test_split,
    get_user_relevant_test_items,
)
from src.metrics import rmse, precision_at_k, recall_at_k, ndcg_at_k
from src.models.popularity import AverageRatingBaseline, PopularityBaseline
from src.models.knn import ItemKNNRecommender
from src.models.mf import PyTorchMatrixFactorization


def evaluate_ranking_metrics(
    model,
    test_user_relevants: Dict[int, set],
    k: int = 10,
    max_eval_users: int = 500,
) -> Tuple[float, float, float]:
    """
    Computes mean Precision@K, Recall@K, and NDCG@K across evaluated users.
    """
    # Sample or select users who have at least one relevant item in test
    eval_users = [
        u for u, items in test_user_relevants.items() if len(items) > 0
    ][:max_eval_users]

    if not eval_users:
        return 0.0, 0.0, 0.0

    precisions = []
    recalls = []
    ndcgs = []

    user_recommendations = model.recommend_for_users(eval_users, k=k, exclude_seen=True)

    for uid in eval_users:
        recs = user_recommendations[uid]
        relevant = test_user_relevants[uid]

        precisions.append(precision_at_k(recs, relevant, k=k))
        recalls.append(recall_at_k(recs, relevant, k=k))
        ndcgs.append(ndcg_at_k(recs, relevant, k=k))

    return (
        float(np.mean(precisions)),
        float(np.mean(recalls)),
        float(np.mean(ndcgs)),
    )


def run_benchmark(
    data_path: str = None,
    k: int = 10,
    min_rating_threshold: float = 3.5,
    max_eval_users: int = 500,
    mf_embedding_dims: List[int] = [16, 32, 64],
    mf_epochs: int = 8,
) -> pd.DataFrame:
    print("=" * 60)
    print("Movie Recommender Benchmark")
    print("=" * 60)

    print("Loading data and creating temporal train/test split...")
    df = load_ratings(data_path)
    print(f"Loaded {len(df):,} ratings ({df['userId'].nunique():,} users, {df['movieId'].nunique():,} movies).")

    train_df, test_df = temporal_train_test_split(df, test_ratio=0.20, min_ratings=5)
    print(f"Train set: {len(train_df):,} ratings | Test set: {len(test_df):,} ratings")

    test_user_relevants = get_user_relevant_test_items(test_df, min_rating=min_rating_threshold)
    y_test_true = test_df["rating"].values

    results = []

    # 1. Baseline: Average Rating + Popularity
    print("\nEvaluating baselines (Average Rating & Most Popular)...")
    t0 = time.time()
    avg_model = AverageRatingBaseline().fit(train_df)
    pop_model = PopularityBaseline().fit(train_df)

    avg_preds = avg_model.predict_batch(test_df)
    baseline_rmse = rmse(y_test_true, avg_preds)
    p_k, r_k, n_k = evaluate_ranking_metrics(
        pop_model, test_user_relevants, k=k, max_eval_users=max_eval_users
    )
    results.append({
        "Model": "Baseline (Avg Rating / Popularity)",
        "RMSE": f"{baseline_rmse:.4f}",
        f"Precision@{k}": f"{p_k:.4f}",
        f"Recall@{k}": f"{r_k:.4f}",
        f"NDCG@{k}": f"{n_k:.4f}",
        "Train Time (s)": f"{time.time() - t0:.2f}",
    })

    # 2. Item-KNN
    print("Training and evaluating Item-KNN...")
    t0 = time.time()
    knn_model = ItemKNNRecommender(n_neighbors=20, metric="cosine").fit(train_df)
    knn_preds = knn_model.predict_batch(test_df)
    knn_rmse = rmse(y_test_true, knn_preds)
    knn_p, knn_r, knn_n = evaluate_ranking_metrics(
        knn_model, test_user_relevants, k=k, max_eval_users=max_eval_users
    )
    results.append({
        "Model": "Item-KNN (k=20, Cosine)",
        "RMSE": f"{knn_rmse:.4f}",
        f"Precision@{k}": f"{knn_p:.4f}",
        f"Recall@{k}": f"{knn_r:.4f}",
        f"NDCG@{k}": f"{knn_n:.4f}",
        "Train Time (s)": f"{time.time() - t0:.2f}",
    })

    # 3. PyTorch Matrix Factorization with Embedding Dim Comparisons
    print("Training PyTorch Matrix Factorization...")
    for dim in mf_embedding_dims:
        print(f" -> MF (dim={dim}, epochs={mf_epochs})...")
        t0 = time.time()
        mf_model = PyTorchMatrixFactorization(
            embedding_dim=dim, epochs=mf_epochs, lr=0.005, weight_decay=1e-4
        ).fit(train_df, verbose=False)

        mf_preds = mf_model.predict_batch(test_df)
        mf_rmse = rmse(y_test_true, mf_preds)
        mf_p, mf_r, mf_n = evaluate_ranking_metrics(
            mf_model, test_user_relevants, k=k, max_eval_users=max_eval_users
        )
        results.append({
            "Model": f"PyTorch MF (dim={dim})",
            "RMSE": f"{mf_rmse:.4f}",
            f"Precision@{k}": f"{mf_p:.4f}",
            f"Recall@{k}": f"{mf_r:.4f}",
            f"NDCG@{k}": f"{mf_n:.4f}",
            "Train Time (s)": f"{time.time() - t0:.2f}",
        })

    # Summary Table
    print("\nBenchmark Results:")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    
    # Custom markdown formatting without tabulate dependency
    headers = list(results_df.columns)
    col_widths = {c: max(len(c), max(len(str(val)) for val in results_df[c])) for c in headers}
    header_line = "| " + " | ".join(c.ljust(col_widths[c]) for c in headers) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[c] for c in headers) + "-|"
    data_lines = [
        "| " + " | ".join(str(row[c]).ljust(col_widths[c]) for c in headers) + " |"
        for _, row in results_df.iterrows()
    ]
    markdown_table = "\n".join([header_line, sep_line] + data_lines)
    print(markdown_table)
    print("=" * 65)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate movie recommender models")
    parser.add_argument("--data", type=str, default=None, help="Path to ratings CSV")
    parser.add_argument("--k", type=int, default=10, help="Top-K recommendations")
    parser.add_argument("--users", type=int, default=500, help="Max users to evaluate for ranking")
    args = parser.parse_args()

    run_benchmark(data_path=args.data, k=args.k, max_eval_users=args.users)

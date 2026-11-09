from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


class ItemKNNRecommender:
    """
    Item-Item Collaborative Filtering using cosine similarity on user interaction vectors.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        min_similarity: float = 0.0,
        metric: str = "cosine",
    ):
        self.n_neighbors = n_neighbors
        self.min_similarity = min_similarity
        self.metric = metric

        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self.item_user_matrix: Optional[csr_matrix] = None
        self.user_item_matrix: Optional[csr_matrix] = None

        self.neighbor_distances: Optional[np.ndarray] = None
        self.neighbor_indices: Optional[np.ndarray] = None

        self.global_mean: float = 3.5
        self.item_means: Dict[int, float] = {}
        self.train_user_items: Dict[int, Set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> "ItemKNNRecommender":
        self.global_mean = float(train_df["rating"].mean())
        self.item_means = train_df.groupby("movieId")["rating"].mean().to_dict()

        unique_users = sorted(train_df["userId"].unique())
        unique_items = sorted(train_df["movieId"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for i, u in enumerate(unique_users)}
        self.item2idx = {m: i for i, m in enumerate(unique_items)}
        self.idx2item = {i: m for i, m in enumerate(unique_items)}

        self.train_user_items = (
            train_df.groupby("userId")["movieId"]
            .apply(lambda x: set(x.tolist()))
            .to_dict()
        )

        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        rows = train_df["movieId"].map(self.item2idx).values
        cols = train_df["userId"].map(self.user2idx).values
        data = train_df["rating"].values

        self.item_user_matrix = csr_matrix(
            (data, (rows, cols)), shape=(n_items, n_users), dtype=np.float32
        )
        self.user_item_matrix = self.item_user_matrix.T.tocsr()

        k_val = min(self.n_neighbors + 1, n_items)
        nn_model = NearestNeighbors(
            n_neighbors=k_val,
            metric=self.metric,
            algorithm="brute",
            n_jobs=-1,
        )
        nn_model.fit(self.item_user_matrix)

        self.neighbor_distances, self.neighbor_indices = nn_model.kneighbors(self.item_user_matrix)
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts rating based on user ratings of the item's nearest neighbors.
        """
        if movie_id not in self.item2idx:
            return self.global_mean
        if user_id not in self.user2idx:
            return self.item_means.get(movie_id, self.global_mean)

        item_idx = self.item2idx[movie_id]
        user_idx = self.user2idx[user_id]

        user_row = self.user_item_matrix.getrow(user_idx)
        user_rated_items = set(user_row.indices)

        distances = self.neighbor_distances[item_idx]
        indices = self.neighbor_indices[item_idx]

        sim_sum = 0.0
        weighted_rating_sum = 0.0

        for dist, neighbor_idx in zip(distances, indices):
            if neighbor_idx == item_idx:
                continue

            sim = max(0.0, 1.0 - dist)
            if sim < self.min_similarity:
                continue

            if neighbor_idx in user_rated_items:
                user_rating = float(user_row[0, neighbor_idx])
                weighted_rating_sum += sim * user_rating
                sim_sum += sim

        if sim_sum > 0:
            return float(weighted_rating_sum / sim_sum)
        return self.item_means.get(movie_id, self.global_mean)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.predict_rating(row.userId, row.movieId) for row in test_df.itertuples()],
            dtype=float,
        )

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        """
        Generates top-K recommendations by aggregating similarities to movies the user liked.
        """
        if user_id not in self.user2idx:
            return list(self.idx2item.values())[:k]

        user_idx = self.user2idx[user_id]
        seen_items = self.train_user_items.get(user_id, set()) if exclude_seen else set()

        user_ratings_row = self.user_item_matrix.getrow(user_idx)
        rated_item_indices = user_ratings_row.indices
        user_ratings = user_ratings_row.data

        favorable_mask = user_ratings >= 3.0
        seed_indices = rated_item_indices[favorable_mask]
        seed_weights = user_ratings[favorable_mask]

        if len(seed_indices) == 0:
            seed_indices = rated_item_indices
            seed_weights = user_ratings

        if len(seed_indices) == 0:
            return list(self.idx2item.values())[:k]

        candidate_scores: Dict[int, float] = {}

        for seed_idx, weight in zip(seed_indices, seed_weights):
            distances = self.neighbor_distances[seed_idx]
            indices = self.neighbor_indices[seed_idx]

            for dist, neighbor_idx in zip(distances, indices):
                neighbor_movie_id = self.idx2item[neighbor_idx]
                if neighbor_movie_id in seen_items:
                    continue

                sim = max(0.0, 1.0 - dist)
                if sim > 0:
                    candidate_scores[neighbor_movie_id] = (
                        candidate_scores.get(neighbor_movie_id, 0.0) + sim * weight
                    )

        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )

        recommended = [m for m, _ in sorted_candidates[:k]]

        if len(recommended) < k:
            for m in sorted(self.item_means, key=self.item_means.get, reverse=True):
                if m not in seen_items and m not in recommended:
                    recommended.append(m)
                if len(recommended) == k:
                    break

        return recommended

    def recommend_for_users(
        self,
        user_ids: List[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[int]]:
        return {uid: self.recommend(uid, k=k, exclude_seen=exclude_seen) for uid in user_ids}

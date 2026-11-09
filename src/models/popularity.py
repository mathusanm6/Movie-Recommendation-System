from typing import Dict, List, Set
import numpy as np
import pandas as pd


class AverageRatingBaseline:
    """
    Predicts ratings using item average rating, falling back to global mean.
    """

    def __init__(self):
        self.global_mean: float = 0.0
        self.item_means: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "AverageRatingBaseline":
        self.global_mean = float(train_df["rating"].mean())
        self.item_means = train_df.groupby("movieId")["rating"].mean().to_dict()
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        return self.item_means.get(movie_id, self.global_mean)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [self.predict_rating(row.userId, row.movieId) for row in test_df.itertuples()],
            dtype=float,
        )


class PopularityBaseline:
    """
    Ranks items by training rating frequency, filtering out items the user has already rated.
    """

    def __init__(self):
        self.popular_movies_ranked: List[int] = []
        self.train_user_items: Dict[int, Set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityBaseline":
        counts = train_df.groupby("movieId")["rating"].count().sort_values(ascending=False)
        self.popular_movies_ranked = counts.index.tolist()
        self.train_user_items = (
            train_df.groupby("userId")["movieId"]
            .apply(lambda x: set(x.tolist()))
            .to_dict()
        )
        return self

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
        seen = self.train_user_items.get(user_id, set()) if exclude_seen else set()
        recs = []
        for movie_id in self.popular_movies_ranked:
            if movie_id not in seen:
                recs.append(movie_id)
            if len(recs) == k:
                break
        return recs

    def recommend_for_users(
        self, user_ids: List[int], k: int = 10, exclude_seen: bool = True
    ) -> Dict[int, List[int]]:
        return {uid: self.recommend(uid, k=k, exclude_seen=exclude_seen) for uid in user_ids}

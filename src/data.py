import os
from typing import Dict, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_ratings(path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads rating records from CSV into a DataFrame with typed columns.
    """
    if path is None:
        path_1m = os.path.join(DATA_DIR, "ratings_1m.csv")
        path_small = os.path.join(DATA_DIR, "ratings.csv")
        path = path_1m if os.path.exists(path_1m) else path_small

    if not os.path.exists(path):
        raise FileNotFoundError(f"Ratings file not found at {path}")

    df = pd.read_csv(path)
    expected_cols = {"userId", "movieId", "rating", "timestamp"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns: {expected_cols - set(df.columns)}")

    df = df[["userId", "movieId", "rating", "timestamp"]].copy()
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def load_movies(path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads movie metadata.
    """
    if path is None:
        path_1m = os.path.join(DATA_DIR, "movies_1m.csv")
        path_small = os.path.join(DATA_DIR, "movies_metadata.csv")
        path = path_1m if os.path.exists(path_1m) else path_small

    if not os.path.exists(path):
        raise FileNotFoundError(f"Movies metadata not found at {path}")

    return pd.read_csv(path)


def temporal_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.20, min_ratings: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a per-user temporal split. The oldest (1 - test_ratio) ratings go to train,
    and the newest test_ratio ratings go to test.
    """
    df = df.sort_values(by=["userId", "timestamp"]).reset_index(drop=True)

    df["user_rating_rank"] = df.groupby("userId").cumcount()
    user_counts = df.groupby("userId")["userId"].transform("count")
    df["user_total_ratings"] = user_counts

    is_test = (df["user_total_ratings"] >= min_ratings) & (
        df["user_rating_rank"] >= np.floor(df["user_total_ratings"] * (1.0 - test_ratio))
    )

    train_df = df[~is_test].drop(columns=["user_rating_rank", "user_total_ratings"]).reset_index(drop=True)
    test_df = df[is_test].drop(columns=["user_rating_rank", "user_total_ratings"]).reset_index(drop=True)

    return train_df, test_df


def build_mappings(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Generates contiguous zero-indexed mappings for user and movie IDs.
    """
    unique_users = sorted(df["userId"].unique())
    unique_items = sorted(df["movieId"].unique())

    user2idx = {u: i for i, u in enumerate(unique_users)}
    idx2user = {i: u for i, u in enumerate(unique_users)}
    item2idx = {m: i for i, m in enumerate(unique_items)}
    idx2item = {i: m for i, m in enumerate(unique_items)}

    return user2idx, idx2user, item2idx, idx2item


def create_user_item_matrix(
    df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
) -> csr_matrix:
    """
    Builds a sparse CSR matrix of user-item ratings (users x items).
    """
    valid_rows = df[df["userId"].isin(user2idx) & df["movieId"].isin(item2idx)]

    row_indices = valid_rows["userId"].map(user2idx).values
    col_indices = valid_rows["movieId"].map(item2idx).values
    ratings = valid_rows["rating"].values

    return csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32,
    )


def get_user_relevant_test_items(
    test_df: pd.DataFrame, min_rating: float = 3.5
) -> Dict[int, Set[int]]:
    """
    Returns a dictionary mapping userId to the set of movieIds rated >= min_rating.
    """
    relevant_df = test_df[test_df["rating"] >= min_rating]
    return (
        relevant_df.groupby("userId")["movieId"]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )


def get_user_train_items(train_df: pd.DataFrame) -> Dict[int, Set[int]]:
    """
    Returns a dictionary mapping userId to the set of movieIds seen in training.
    """
    return (
        train_df.groupby("userId")["movieId"]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RatingsDataset(Dataset):
    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, ratings: np.ndarray):
        self.users = torch.tensor(user_indices, dtype=torch.long)
        self.items = torch.tensor(item_indices, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MatrixFactorizationModule(nn.Module):
    """
    Matrix Factorization with user/item embeddings and explicit bias terms:
    pred(u, i) = global_mean + b_u + b_i + <p_u, q_i>
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32, global_mean: float = 3.5):
        super().__init__()
        self.global_mean = nn.Parameter(torch.tensor(global_mean, dtype=torch.float32), requires_grad=False)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)

        dot = (u_emb * i_emb).sum(dim=-1, keepdim=True)
        u_b = self.user_bias(user_idx)
        i_b = self.item_bias(item_idx)

        pred = self.global_mean + u_b + i_b + dot
        return pred.squeeze(-1)


class PyTorchMatrixFactorization:
    """
    Matrix Factorization model trained via MSE loss with Adam.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        epochs: int = 8,
        batch_size: int = 1024,
        device: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[MatrixFactorizationModule] = None
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self.global_mean: float = 3.5
        self.item_means: Dict[int, float] = {}
        self.train_user_items: Dict[int, Set[int]] = {}

    def fit(self, train_df: pd.DataFrame, verbose: bool = False) -> "PyTorchMatrixFactorization":
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

        user_indices = train_df["userId"].map(self.user2idx).values
        item_indices = train_df["movieId"].map(self.item2idx).values
        ratings = train_df["rating"].values

        dataset = RatingsDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MatrixFactorizationModule(
            num_users=len(self.user2idx),
            num_items=len(self.item2idx),
            embedding_dim=self.embedding_dim,
            global_mean=self.global_mean,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_samples = 0
            for u_b, i_b, r_b in dataloader:
                u_b = u_b.to(self.device)
                i_b = i_b.to(self.device)
                r_b = r_b.to(self.device)

                optimizer.zero_grad()
                pred = self.model(u_b, i_b)
                loss = criterion(pred, r_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(r_b)
                total_samples += len(r_b)

            epoch_rmse = np.sqrt(total_loss / total_samples)
            if verbose and (epoch % 2 == 0 or epoch == self.epochs):
                print(f"Epoch {epoch}/{self.epochs} - Loss: {epoch_rmse:.4f}")

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if movie_id not in self.item2idx:
            return self.global_mean
        if user_id not in self.user2idx:
            return self.item_means.get(movie_id, self.global_mean)

        self.model.eval()
        with torch.no_grad():
            u_t = torch.tensor([self.user2idx[user_id]], dtype=torch.long, device=self.device)
            i_t = torch.tensor([self.item2idx[movie_id]], dtype=torch.long, device=self.device)
            pred = self.model(u_t, i_t).item()
            return float(np.clip(pred, 0.5, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction for entire test dataframe.
        """
        self.model.eval()
        users = test_df["userId"].values
        items = test_df["movieId"].values

        known_mask = np.array([u in self.user2idx and m in self.item2idx for u, m in zip(users, items)])
        predictions = np.empty(len(test_df), dtype=float)

        for idx in np.where(~known_mask)[0]:
            m = items[idx]
            predictions[idx] = self.item_means.get(m, self.global_mean)

        known_indices = np.where(known_mask)[0]
        if len(known_indices) > 0:
            known_u = [self.user2idx[users[i]] for i in known_indices]
            known_i = [self.item2idx[items[i]] for i in known_indices]

            with torch.no_grad():
                u_t = torch.tensor(known_u, dtype=torch.long, device=self.device)
                i_t = torch.tensor(known_i, dtype=torch.long, device=self.device)
                preds = self.model(u_t, i_t).cpu().numpy()
                predictions[known_indices] = np.clip(preds, 0.5, 5.0)

        return predictions

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        if user_id not in self.user2idx:
            return sorted(self.item_means, key=self.item_means.get, reverse=True)[:k]

        self.model.eval()
        user_idx = self.user2idx[user_id]
        seen_items = self.train_user_items.get(user_id, set()) if exclude_seen else set()

        with torch.no_grad():
            u_t = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            u_emb = self.model.user_embedding(u_t)
            u_b = self.model.user_bias(u_t).squeeze()

            all_i_emb = self.model.item_embedding.weight
            all_i_b = self.model.item_bias.weight.squeeze()

            scores = (
                self.model.global_mean
                + u_b
                + all_i_b
                + torch.matmul(all_i_emb, u_emb.squeeze(0))
            ).cpu().numpy()

        for m in seen_items:
            if m in self.item2idx:
                scores[self.item2idx[m]] = -np.inf

        top_indices = np.argpartition(scores, -k)[-k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [self.idx2item[idx] for idx in sorted_top_indices]

    def recommend_for_users(
        self,
        user_ids: List[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[int, List[int]]:
        return {uid: self.recommend(uid, k=k, exclude_seen=exclude_seen) for uid in user_ids}

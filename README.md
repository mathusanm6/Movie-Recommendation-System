# Movie Recommendation System

This project evaluates movie recommendation models on the MovieLens dataset using a realistic temporal train/test split. It compares heuristic baselines against Item-Item Collaborative Filtering (KNN) and a PyTorch Matrix Factorization model across both rating accuracy (RMSE) and top-K ranking metrics (Precision@10, Recall@10, NDCG@10).

## Project Structure

```text
├── README.md
├── recommendation-algorithms-netflix.pdf  # Presentation slides that started this project
├── requirements.txt
├── data/
│   ├── ratings.csv         # Raw ratings
│   └── movies_metadata.csv # Movie titles
├── src/
│   ├── data.py             # Temporal train/test split and sparse matrices
│   ├── metrics.py          # RMSE, Precision@K, Recall@K, NDCG@K
│   ├── evaluate.py         # Benchmark runner
│   └── models/
│       ├── popularity.py   # Average rating & most popular baselines
│       ├── knn.py          # Item-Item Collaborative Filtering
│       └── mf.py           # PyTorch Matrix Factorization
├── tests/
│   └── test_metrics.py     # Unit tests for metrics
└── notebooks/
    └── analysis.ipynb      # Cold-start and popularity analysis
```

## Evaluation Setup

### 1. Temporal Split
Randomly splitting ratings causes data leakage because future ratings are used to predict past ones. In `src/data.py`, each user's ratings are sorted chronologically by timestamp:
- **Train (80%)**: The user's older ratings.
- **Test (20%)**: The user's newest ratings.

### 2. Metrics
- **RMSE**: Evaluates rating prediction error against true ratings.
- **Precision@10**: Fraction of top-10 recommended movies that the user rated favorably (rating >= 3.5) in the test set.
- **Recall@10**: Fraction of the user's favorable test movies captured in the top 10.
- **NDCG@10**: Ranking metric that gives higher credit to relevant movies ranked near the top of the recommendation list.

## Benchmark Results

Results evaluated on the temporal test set:

| Model | RMSE | Precision@10 | Recall@10 | NDCG@10 | Train Time (s) |
|---|---|---|---|---|---|
| Baseline (Avg Rating / Popularity) | 1.0227 | 0.0600 | 0.0495 | 0.0798 | 0.02 |
| Item-KNN (k=20, Cosine) | 1.0209 | 0.0850 | 0.0891 | 0.1042 | 1.79 |
| PyTorch MF (dim=16) | 0.9168 | 0.0480 | 0.0351 | 0.0578 | 5.78 |
| PyTorch MF (dim=32) | 0.9200 | 0.0520 | 0.0372 | 0.0643 | 3.78 |
| PyTorch MF (dim=64) | 0.9243 | 0.0490 | 0.0361 | 0.0594 | 4.02 |

### Notes on the Results
- **Ranking vs. Rating Prediction**: Item-KNN achieved the highest NDCG@10 (0.1042), beating the popularity baseline (0.0798). PyTorch Matrix Factorization achieved the best RMSE (0.9168), but lower ranking metrics. This is expected: minimizing Mean Squared Error optimizes for accuracy across all ratings (including mediocre films), whereas Item-KNN recommends directly from items the user rated highly.
- **Embedding Dimensions**: For Matrix Factorization, smaller latent dimensions (dim=16 and dim=32) performed better than dim=64 on this dataset, as larger embeddings began to slightly overfit.

## Context

This project originated from a university course presentation on recommendation algorithms and the Netflix Prize. The slides from that presentation are included in [recommendation-algorithms-netflix.pdf](recommendation-algorithms-netflix.pdf).

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Unit Tests
```bash
pytest
```

### Run the Benchmark
```bash
python src/evaluate.py
```

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Load the data
user_ratings_df = pd.read_csv('data/ratings.csv')

movie_metadata = pd.read_csv('data/movies_metadata.csv')

# Merge the data
movie_data = user_ratings_df.merge(movie_metadata, on='movieId')

# Create a user-item matrix
user_item_matrix = user_ratings_df.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)

# Create a KNN model
cf_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)

# Fit the model
cf_knn_model.fit(user_item_matrix)
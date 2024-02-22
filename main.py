import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
user_ratings_df = pd.read_csv('data/ratings.csv')

movie_metadata = pd.read_csv('data/movies_metadata.csv')

# Merge the data
movie_data = user_ratings_df.merge(movie_metadata, on='movieId')

# Create a user-item matrix
user_item_matrix = user_ratings_df.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)

print(user_item_matrix.head())
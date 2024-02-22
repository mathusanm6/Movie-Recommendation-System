import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
user_ratings_df = pd.read_csv('data/ratings.csv')

movie_metadata = pd.read_csv('data/movies_metadata.csv')

# Merge the data
movie_data = user_ratings_df.merge(movie_metadata, on='movieId')
print(movie_data.head())
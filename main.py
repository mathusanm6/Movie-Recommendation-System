import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
user_ratings_df = pd.read_csv('data/ratings.csv')

movie_metadata_df = pd.read_csv('data/movies_metadata.csv')
movie_metadata_df = movie_metadata_df[['title', 'genres']]
print(movie_metadata_df.head())
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Load the data
user_ratings_df = pd.read_csv("data/ratings.csv")

movie_metadata = pd.read_csv("data/movies_metadata.csv")

# Merge the data
movie_data = user_ratings_df.merge(movie_metadata, on="movieId")

# Create a user-item matrix
user_item_matrix = user_ratings_df.pivot(
    index=["movieId"], columns=["userId"], values="rating"
).fillna(0)

# Create a KNN model
cf_model = NearestNeighbors(
    metric="cosine", algorithm="brute", n_neighbors=10, n_jobs=-1
)


def movie_recommender_engine(movie_name, matrix, cf_model, n_recs):
    # Fit the model
    cf_knn_model = cf_model.fit(user_item_matrix)

    # Extract the movie index
    movie_id = process.extractOne(movie_name, movie_metadata["title"])[2]

    # Calculate neighbor distances
    distances, indices = cf_knn_model.kneighbors(matrix.loc[movie_id, :].values.reshape(1, -1), n_neighbors=n_recs)
    movie_rec_ids = sorted(
        list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
        key=lambda x: x[1],
    )[:0:-1]

    # List to store recommendations
    cf_recs = []
    for i in movie_rec_ids:
        cf_recs.append({"Title": movie_metadata["title"][i[0]], "Distance": i[1]})

    # Select top number of recommendations needed
    df = pd.DataFrame(cf_recs, index=range(1, n_recs))

    return df


n_recs = 10 # Number of recommendations (excluding the movie itself)
print(movie_recommender_engine("Terminator", user_item_matrix, cf_model, n_recs))

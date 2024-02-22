import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
user_ratings_df = pd.read_csv('data/ratings.csv')
print(user_ratings_df.head())   # print the first 5 rows of the dataframe
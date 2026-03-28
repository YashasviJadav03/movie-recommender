"""
Configuration file for the movie recommendation system.
Centralized parameters for easy tuning and consistency.
"""

# Model Parameters
SVD_CONFIG = {
    "n_factors": 50,
    "n_epochs": 30,
    "lr_all": 0.005,
    "reg_all": 0.05,
    "random_state": 42
}

# Hybrid Model Weights
HYBRID_WEIGHTS = {
    "movie_content": 0.7,
    "movie_popularity": 0.3,
    "user_cf": 0.6,
    "user_content": 0.4
}

# Recommendation Parameters
LIKED_THRESHOLD = 4.0  # Minimum rating to consider a movie "liked"
DEFAULT_TOP_N = 10
MAX_TOP_N = 20
MIN_TOP_N = 5

# Data Paths
DATA_DIR = "data/ml-100k"
PROCESSED_DIR = "data/processed"

# User Constraints
MIN_USER_ID = 1
MAX_USER_ID = 943  # MovieLens 100K has 943 users

# Search Parameters
FUZZY_SEARCH_MIN_SCORE = 60
FUZZY_SEARCH_LIMIT = 15

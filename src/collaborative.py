import pandas as pd
import numpy as np

try:
    from surprise import Dataset, Reader, SVD
except ImportError:
    Dataset = None
    Reader = None
    SVD = None

from config import SVD_CONFIG


def train_svd_model(ratings, **kwargs):
    """
    Train SVD collaborative filtering model.
    
    Args:
        ratings: DataFrame with userId, movieId, rating columns
        **kwargs: Override default SVD_CONFIG parameters
        
    Returns:
        Trained SVD model or None if surprise is not available
    """
    if SVD is None:
        return None
    
    # Merge config with any overrides
    config = {**SVD_CONFIG, **kwargs}
    
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    
    model = SVD(
        n_factors=config["n_factors"],
        n_epochs=config["n_epochs"],
        lr_all=config["lr_all"],
        reg_all=config["reg_all"],
        random_state=config["random_state"]
    )
    
    model.fit(trainset)
    return model


def predict_user_ratings(user_id, movie_ids, cf_model):
    """
    Predict ratings for a user on multiple movies (vectorized).
    
    Args:
        user_id: User ID
        movie_ids: Array or list of movie IDs
        cf_model: Trained SVD model
        
    Returns:
        numpy array of predicted ratings
    """
    if cf_model is None:
        return np.zeros(len(movie_ids))
    
    predictions = np.array([cf_model.predict(user_id, mid).est for mid in movie_ids])
    return predictions


def get_user_recommendations(user_id, ratings, movies, cf_model, top_n=10):
    """
    Get collaborative filtering recommendations for a user.
    
    Args:
        user_id: User ID
        ratings: Ratings DataFrame
        movies: Movies DataFrame
        cf_model: Trained SVD model
        top_n: Number of recommendations
        
    Returns:
        DataFrame with movieId, title, score columns
    """
    if cf_model is None:
        return pd.DataFrame(columns=["movieId", "title", "score"])
    
    # Get movies the user hasn't seen
    seen_movies = set(ratings.loc[ratings["userId"] == user_id, "movieId"].values)
    unseen_movies = movies[~movies["movieId"].isin(seen_movies)].copy()
    
    if unseen_movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])
    
    # Vectorized prediction
    unseen_movies["score"] = predict_user_ratings(
        user_id, 
        unseen_movies["movieId"].values, 
        cf_model
    )
    
    # Sort and return top N
    recommendations = unseen_movies.nlargest(top_n, "score")[["movieId", "title", "score"]]
    
    return recommendations.reset_index(drop=True)


def validate_user_exists(user_id, ratings):
    """Check if user exists in ratings data."""
    return user_id in ratings["userId"].values

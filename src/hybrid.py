import pandas as pd
import numpy as np

from config import HYBRID_WEIGHTS, LIKED_THRESHOLD


def minmax_normalize(series):
    """Normalize a pandas Series to 0-1 range."""
    min_val = float(series.min())
    max_val = float(series.max())
    
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def get_hybrid_movie_recommendations(
    title, 
    movies_content, 
    cosine_sim, 
    popularity, 
    top_n=10,
    content_weight=None,
    popularity_weight=None
):
    """
    Get hybrid recommendations combining content similarity and popularity.
    
    Args:
        title: Movie title
        movies_content: DataFrame with movie data
        cosine_sim: Precomputed similarity matrix
        popularity: DataFrame with movieId and count columns
        top_n: Number of recommendations
        content_weight: Weight for content similarity (default from config)
        popularity_weight: Weight for popularity (default from config)
        
    Returns:
        DataFrame with movieId, title, score columns
    """
    from src.content import get_content_recommendations
    
    # Use config defaults if not specified
    if content_weight is None:
        content_weight = HYBRID_WEIGHTS["movie_content"]
    if popularity_weight is None:
        popularity_weight = HYBRID_WEIGHTS["movie_popularity"]
    
    # Get content-based recommendations (larger pool)
    content_recs = get_content_recommendations(title, movies_content, cosine_sim, top_n=200)
    
    if content_recs.empty:
        return content_recs
    
    # Add popularity scores
    pop_dict = popularity.set_index("movieId")["count"].to_dict()
    content_recs["popularity"] = content_recs["movieId"].map(pop_dict).fillna(0.0)
    
    # Combine scores with weights
    content_recs["score"] = (
        content_weight * minmax_normalize(content_recs["score"]) +
        popularity_weight * minmax_normalize(content_recs["popularity"])
    )
    
    # Sort and return top N
    result = content_recs.nlargest(top_n, "score")[["movieId", "title", "score"]]
    return result.reset_index(drop=True)


def get_hybrid_user_recommendations(
    user_id,
    ratings,
    movies,
    cosine_sim,
    movieid_to_index,
    cf_model,
    top_n=10,
    cf_weight=None,
    content_weight=None,
    liked_threshold=None
):
    """
    Get hybrid recommendations for a user combining CF and content.
    
    Args:
        user_id: User ID
        ratings: Ratings DataFrame
        movies: Movies DataFrame
        cosine_sim: Precomputed similarity matrix
        movieid_to_index: Mapping from movieId to matrix index
        cf_model: Trained collaborative filtering model
        top_n: Number of recommendations
        cf_weight: Weight for CF score (default from config)
        content_weight: Weight for content score (default from config)
        liked_threshold: Rating threshold for liked movies (default from config)
        
    Returns:
        DataFrame with movieId, title, score columns
    """
    from src.collaborative import predict_user_ratings, validate_user_exists
    
    # Use config defaults if not specified
    if cf_weight is None:
        cf_weight = HYBRID_WEIGHTS["user_cf"]
    if content_weight is None:
        content_weight = HYBRID_WEIGHTS["user_content"]
    if liked_threshold is None:
        liked_threshold = LIKED_THRESHOLD
    
    if cf_model is None or not validate_user_exists(user_id, ratings):
        return pd.DataFrame(columns=["movieId", "title", "score"])
    
    # Get unseen movies
    seen_movies = set(ratings.loc[ratings["userId"] == user_id, "movieId"].values)
    candidates = movies[~movies["movieId"].isin(seen_movies)].copy()
    
    if candidates.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])
    
    # Get CF predictions (vectorized)
    candidates["cf_score"] = predict_user_ratings(
        user_id,
        candidates["movieId"].values,
        cf_model
    )
    
    # Get content scores based on user's liked movies
    liked_movies = ratings[
        (ratings["userId"] == user_id) & 
        (ratings["rating"] >= liked_threshold)
    ]["movieId"].unique()
    
    liked_indices = [movieid_to_index[mid] for mid in liked_movies if mid in movieid_to_index]
    
    if liked_indices:
        # Average similarity to all liked movies
        avg_similarity = cosine_sim[liked_indices].mean(axis=0)
        candidates["content_score"] = candidates["movieId"].apply(
            lambda mid: float(avg_similarity[movieid_to_index[mid]]) 
            if mid in movieid_to_index else 0.0
        )
    else:
        candidates["content_score"] = 0.0
    
    # Combine scores
    candidates["score"] = (
        cf_weight * minmax_normalize(candidates["cf_score"]) +
        content_weight * minmax_normalize(candidates["content_score"])
    )
    
    # Sort and return top N
    result = candidates.nlargest(top_n, "score")[["movieId", "title", "score"]]
    return result.reset_index(drop=True)

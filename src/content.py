import pandas as pd
import numpy as np


def get_content_recommendations(title, movies_content, cosine_sim, top_n=10):
    """
    Get content-based recommendations for a movie.
    
    Args:
        title: Movie title
        movies_content: DataFrame with movieId, title, genres
        cosine_sim: Precomputed cosine similarity matrix
        top_n: Number of recommendations
        
    Returns:
        DataFrame with movieId, title, score columns
    """
    # Validate movie exists
    matching_movies = movies_content[movies_content["title"] == title]
    if matching_movies.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])
    
    idx = int(matching_movies.index[0])
    
    # Get similarity scores
    sim_scores = pd.Series(cosine_sim[idx], index=movies_content.index)
    sim_scores = sim_scores.sort_values(ascending=False).iloc[1:top_n + 1]
    
    # Build result DataFrame
    result = movies_content.loc[sim_scores.index, ["movieId", "title"]].copy()
    result["score"] = sim_scores.values
    
    return result.reset_index(drop=True)


def get_movie_popularity(ratings):
    """Calculate movie popularity based on rating counts."""
    popularity = ratings["movieId"].value_counts().rename("count").reset_index()
    popularity.columns = ["movieId", "count"]
    return popularity


def validate_movie_exists(title, movies_content):
    """Check if a movie title exists in the dataset."""
    return title in movies_content["title"].values
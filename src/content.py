import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load and prepare movie data."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one directory to the project root
    project_root = os.path.dirname(script_dir)
    # Construct the path to the data file
    data_path = os.path.join(project_root, "data", "ml-100k", "u.item")
    
    movies = pd.read_csv(
        data_path,
        sep="|",
        encoding="latin-1",
        header=None,
        names=[
            "movieId", "title", "release_date", "video_release_date", "IMDb_URL",
            *[f"genre_{i}" for i in range(19)]
        ]
    )
    
    # Extract genres
    genre_columns = movies.columns[5:]
    movies['genres'] = movies[genre_columns].apply(
        lambda x: '|'.join(genre_columns[x == 1]), axis=1
    )
    
    return movies

# Rest of the code remains the same
movies = load_data()
title_to_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_content(movie_title, top_n=10):
    """
    Get movie recommendations based on content similarity.
    
    Args:
        movie_title (str): Title of the movie to get recommendations for
        top_n (int): Number of recommendations to return
        
    Returns:
        pandas.DataFrame: DataFrame of recommended movies with columns:
            - movieId
            - title
            - genres
            - similarity_score
    """
    if movie_title not in title_to_index:
        return f"Movie '{movie_title}' not found."

    idx = title_to_index[movie_title]

    # Get similarity scores with index
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first movie (itself) and get top_n
    sim_scores = sim_scores[1:top_n + 1]

    # Get movie indices and similarity scores
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    # Create result DataFrame
    result = movies.loc[movie_indices, ["movieId", "title", "genres"]]
    result["similarity_score"] = similarity_scores
    
    return result

if __name__ == "__main__":
    # Example usage
    print("Recommendations for 'Toy Story (1995)'")
    print(recommend_content("Toy Story (1995)"))
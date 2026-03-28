import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_raw_data():
    """Load raw MovieLens 100K data."""
    root = get_project_root()
    data_dir = root / "data" / "ml-100k"
    
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        names=[
            "movieId", "title", "release_date", "video_release_date", "IMDb_URL",
            *[f"genre_{i}" for i in range(19)]
        ]
    )
    
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"]
    )
    
    return movies, ratings


def process_movies(movies):
    """Process movie data and extract genres."""
    genre_columns = movies.columns[5:24]
    movies['genres'] = movies[genre_columns].apply(
        lambda x: ' '.join([col.replace('genre_', '') for col, val in zip(genre_columns, x) if val == 1]),
        axis=1
    )
    
    movies_meta = movies[["movieId", "title", "release_date"]].copy()
    movies_content = movies[["movieId", "title", "genres"]].copy()
    
    return movies_meta, movies_content


def compute_content_similarity(movies_content):
    """Compute TF-IDF and cosine similarity matrix."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_content["genres"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim


def clean_ratings(ratings):
    """Clean and validate ratings data."""
    ratings_clean = ratings.copy()
    ratings_clean = ratings_clean.dropna()
    ratings_clean = ratings_clean[
        (ratings_clean["rating"] >= 0.5) & 
        (ratings_clean["rating"] <= 5.0)
    ]
    
    return ratings_clean


def save_processed_data(ratings_clean, movies_meta, movies_content, cosine_sim, output_dir="data/processed"):
    """Save processed data to disk."""
    root = get_project_root()
    output_path = root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    ratings_clean.to_csv(output_path / "ratings_clean.csv", index=False)
    movies_meta.to_csv(output_path / "movies_meta.csv", index=False)
    movies_content.to_csv(output_path / "movies_content.csv", index=False)
    np.save(output_path / "content_cosine_sim.npy", cosine_sim)
    
    print(f"Processed data saved to {output_path}")


def prepare_all_data():
    """Main function to prepare all processed data."""
    print("Loading raw data...")
    movies, ratings = load_raw_data()
    
    print("Processing movies...")
    movies_meta, movies_content = process_movies(movies)
    
    print("Computing content similarity...")
    cosine_sim = compute_content_similarity(movies_content)
    
    print("Cleaning ratings...")
    ratings_clean = clean_ratings(ratings)
    
    print("Saving processed data...")
    save_processed_data(ratings_clean, movies_meta, movies_content, cosine_sim)
    
    print("Data preparation complete!")
    return ratings_clean, movies_meta, movies_content, cosine_sim


if __name__ == "__main__":
    prepare_all_data()

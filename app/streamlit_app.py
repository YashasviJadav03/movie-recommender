import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import process

try:
    from surprise import Dataset, Reader, SVD
except Exception:
    Dataset = None
    Reader = None
    SVD = None


ROOT = Path(__file__).resolve().parents[1]


def processed_dir() -> Path:
    candidates = [
        ROOT / "data" / "processed",
        ROOT / "notebooks" / "data" / "processed",
    ]
    for p in candidates:
        if (p / "ratings_clean.csv").exists():
            return p
    return candidates[0]


@st.cache_data
def load_data():
    pdir = processed_dir()
    ratings_path = pdir / "ratings_clean.csv"
    movies_path = pdir / "movies_meta.csv"
    movies_content_path = pdir / "movies_content.csv"
    cosine_path = pdir / "content_cosine_sim.npy"

    if not ratings_path.exists():
        st.error(
            "Missing processed data. Expected ratings_clean.csv in: "
            + str(pdir)
            + "\n\nRun your preprocessing notebooks/pipeline to generate data/processed."
        )
        st.stop()

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    movies_content = pd.read_csv(movies_content_path)
    cosine_sim = np.load(cosine_path)

    movieid_to_index = pd.Series(movies_content.index, index=movies_content["movieId"]).to_dict()
    titles = movies_content["title"].astype(str).tolist()

    pop = ratings["movieId"].value_counts().rename("count").reset_index()
    pop.columns = ["movieId", "count"]
    movies_pop = movies.merge(pop, on="movieId", how="left", validate="m:1").fillna({"count": 0})
    return ratings, movies, movies_content, cosine_sim, movieid_to_index, titles, movies_pop


@st.cache_resource
def train_cf(ratings: pd.DataFrame):
    if SVD is None:
        return None
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=100, n_epochs=20, random_state=42)
    model.fit(trainset)
    return model


def minmax(s: pd.Series) -> pd.Series:
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)




def show_recs(df: pd.DataFrame):
    """Display movie recommendations in a clean table."""
    if df.empty:
        st.warning("No recommendations found.")
        return
    
    st.subheader("🎬 Recommended Movies")
    
    if 'score' in df.columns:
        st.dataframe(
            df[['title', 'score']],
            column_config={
                "title": "Movie Title",
                "score": st.column_config.NumberColumn("Match Score", format="%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(
            df[['title']],
            column_config={"title": "Movie Title"},
            use_container_width=True,
            hide_index=True
        )
        
        # Always show the data table
        st.subheader("📊 Recommendation Details")
        st.dataframe(
            df[['title', 'score']] if 'score' in df.columns else df[['title']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "title": "Movie Title",
                "score": st.column_config.NumberColumn("Match Score", format="%.2f")
            }
        )


def recommend_content_movie(title: str, movies_content: pd.DataFrame, cosine_sim: np.ndarray, top_n: int):
    idx = int(movies_content.index[movies_content["title"].astype(str) == title][0])
    sims = pd.Series(cosine_sim[idx], index=movies_content.index).sort_values(ascending=False).iloc[1 : top_n + 1]
    out = movies_content.loc[sims.index, ["movieId", "title"]].copy()
    out["score"] = sims.values
    return out


def recommend_movie_hybrid(title: str, movies_content: pd.DataFrame, cosine_sim: np.ndarray, movies_pop: pd.DataFrame, top_n: int):
    content = recommend_content_movie(title, movies_content, cosine_sim, top_n=200)
    pop = movies_pop.set_index("movieId")["count"]
    df = content.copy()
    df["pop"] = df["movieId"].map(pop).fillna(0.0)
    df["score"] = 0.7 * minmax(df["score"]) + 0.3 * minmax(df["pop"])
    return df.sort_values("score", ascending=False).head(top_n)[["movieId", "title", "score"]]


def recommend_user_cf(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame, cf_model, top_n: int):
    if cf_model is None:
        return pd.DataFrame(columns=["movieId", "title", "score"])
    seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"].values)
    candidates = movies.loc[~movies["movieId"].isin(seen), ["movieId", "title"]].copy()
    candidates["score"] = candidates["movieId"].apply(lambda mid: cf_model.predict(user_id, mid).est)
    return candidates.sort_values("score", ascending=False).head(top_n)


def recommend_user_hybrid(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    cosine_sim: np.ndarray,
    movieid_to_index: dict,
    cf_model,
    top_n: int,
):
    if cf_model is None:
        return pd.DataFrame(columns=["movieId", "title", "score"])
    seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"].values)
    candidates = movies.loc[~movies["movieId"].isin(seen), ["movieId", "title"]].copy()
    candidates["cf"] = candidates["movieId"].apply(lambda mid: cf_model.predict(user_id, mid).est)

    liked = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= 4)]["movieId"].unique()
    liked_idx = [movieid_to_index[mid] for mid in liked if mid in movieid_to_index]
    if liked_idx:
        vec = cosine_sim[liked_idx].mean(axis=0)
        candidates["content"] = candidates["movieId"].apply(
            lambda mid: float(vec[movieid_to_index[mid]]) if mid in movieid_to_index else 0.0
        )
    else:
        candidates["content"] = 0.0

    candidates["score"] = 0.6 * minmax(candidates["cf"]) + 0.4 * minmax(candidates["content"])
    return candidates.sort_values("score", ascending=False).head(top_n)[["movieId", "title", "score"]]


def search_movies(query: str, titles: list, min_score: int = 60) -> list:
    """Search for movies using fuzzy matching"""
    if not query:
        return titles[:10]  # Return first 10 if no query
    
    # Get best matches with scores
    matches = process.extract(query, titles, limit=10)
    
    # Filter by score and return only titles
    return [match[0] for match in matches if match[1] >= min_score] or ["No matches found"]


def render_movie_mode(titles, movies_content, cosine_sim, movies_pop, top_n: int, algo: str):
    # Search box
    search_query = st.text_input("Search for a movie...", "")
    
    # Get matching movies
    matching_movies = search_movies(search_query, titles)
    
    # Display dropdown with matching movies
    if matching_movies[0] == "No matches found":
        st.warning("No matching movies found. Try a different search term.")
        return
        
    selected = st.selectbox(
        "Select a movie",
        options=matching_movies,
        index=0
    )

    if algo == "CF":
        st.info("CF needs a user ID. Switch to Hybrid/Content or change input.")
        return

    recs = (
        recommend_movie_hybrid(selected, movies_content, cosine_sim, movies_pop, top_n)
        if algo == "Hybrid"
        else recommend_content_movie(selected, movies_content, cosine_sim, top_n)
    )
    show_recs(recs)


def render_user_mode(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    cosine_sim: np.ndarray,
    movieid_to_index: dict,
    movies_pop: pd.DataFrame,
    cf_model,
    top_n: int,
    algo: str,
):
    if algo == "Content":
        st.info("Content is movie-to-movie. Switch to CF/Hybrid or change input.")
        return

    if user_id not in set(ratings["userId"].values):
        st.info("Cold-start user: showing popular movies.")
        recs = movies_pop.sort_values("count", ascending=False).head(top_n)[["movieId", "title"]].copy()
        recs["score"] = movies_pop.sort_values("count", ascending=False).head(top_n)["count"].values
        show_recs(recs)
        return

    recs = (
        recommend_user_hybrid(user_id, ratings, movies, cosine_sim, movieid_to_index, cf_model, top_n)
        if algo == "Hybrid"
        else recommend_user_cf(user_id, ratings, movies, cf_model, top_n)
    )
    show_recs(recs)


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("🎬 Movie Recommender")

    ratings, movies, movies_content, cosine_sim, movieid_to_index, titles, movies_pop = load_data()
    cf_model = train_cf(ratings)

    if cf_model is None:
        st.sidebar.warning("CF/Hybrid (user-based) needs scikit-surprise. Use Python 3.11 or install a compatible wheel.")

    st.sidebar.header("Controls")
    mode = st.sidebar.radio("Input", ["Movie Title", "User ID"], horizontal=True)
    algo = st.sidebar.radio("Model", ["Content", "CF", "Hybrid"], horizontal=True)
    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

    if mode == "Movie Title":
        render_movie_mode(titles, movies_content, cosine_sim, movies_pop, top_n, algo)
    else:
        user_id = int(st.number_input("User ID", min_value=1, value=1, step=1))
        render_user_mode(
            user_id,
            ratings,
            movies,
            cosine_sim,
            movieid_to_index,
            movies_pop,
            cf_model,
            top_n,
            algo,
        )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()

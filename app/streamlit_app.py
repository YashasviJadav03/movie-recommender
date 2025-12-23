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
    """Display movie recommendations with compact spacing and proportional scores."""
    # Custom CSS for compact layout and better progress bars
    st.markdown("""
    <style>
        /* Reduce space between columns */
        [data-testid="stHorizontalBlock"] {
            gap: 0.25rem !important;
            align-items: center;
        }
        /* Make progress bars more compact */
        .stProgress {
            margin: 0 !important;
        }
        .stProgress > div > div > div > div {
            min-width: 100px !important;
            height: 0.75rem !important;
        }
        /* Reduce space between items */
        .stContainer > div {
            margin: 0.25rem 0 !important;
            padding: 0.25rem 0 !important;
        }
        /* Make dividers more subtle */
        .stDivider {
            margin: 0.25rem 0 !important;
            border-top: 1px solid rgba(250, 250, 250, 0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No recommendations found. Try adjusting your search criteria.")
        return
    
    st.subheader("🎬 Recommended Movies")
    
    if 'score' in df.columns:
        df = df.copy()
        # Normalize scores to be proportional within the current results
        min_score = df['score'].min()
        score_range = df['score'].max() - min_score
        df['progress_value'] = (df['score'] - min_score) / (score_range if score_range > 0 else 1)
        
        # Ensure progress values are between 0.1 and 1.0 for better visibility
        df['progress_value'] = df['progress_value'].clip(0.1, 1.0)
        
        for _, row in df.iterrows():
            with st.container():
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f"**{row['title']}**")
                with cols[1]:
                    st.progress(
                        float(row['progress_value']),
                        text=f"{row['score']:.1f}" if row['score'] < 10 else f"{row['score']:.0f}"
                    )
            # Add a subtle divider
            st.markdown("<div style='margin: 0.1rem 0;'></div>", unsafe_allow_html=True)
    else:
        st.markdown("\n".join([f"- {title}" for title in df['title']]))


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
    """Search for movies using fuzzy matching with improved results"""
    if not query or len(query) < 2:
        return titles[:8]  # Return first 8 if no or very short query
    
    # Get best matches with scores
    matches = process.extract(query, titles, limit=15, scorer=process.fuzz.token_sort_ratio)
    
    # Filter by score and return only titles of high confidence matches
    return [match[0] for match in matches if match[1] >= min_score] or ["No matches found"]


def render_movie_mode(titles, movies_content, cosine_sim, movies_pop, top_n: int, algo: str):
    st.markdown("### 🎥 Find Similar Movies")
    st.markdown("Search for a movie to get personalized recommendations based on your selection.")
    
    # Improved search box with placeholder and help text
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Search for a movie...",
                "",
                placeholder="Type at least 2 characters to search",
                help="Start typing to find movies in our database"
            )
    
    # Get matching movies
    matching_movies = search_movies(search_query, titles)
    
    # Display matching movies in a more user-friendly way
    if matching_movies[0] == "No matches found":
        st.warning("🔍 No matching movies found. Try a different search term.")
        return
    
    # Show popular movies when no search query
    if not search_query or len(search_query) < 2:
        st.info("🌟 Popular Movies (select one to get recommendations)")
    
    # Use radio buttons for better mobile experience
    selected = st.radio(
        "Select a movie:",
        options=matching_movies,
        index=0,
        label_visibility="collapsed"
    )

    if algo == "CF":
        st.info("ℹ️ For user-based recommendations, switch to 'User ID' mode.")
        return

    with st.spinner(f"Finding similar movies to '{selected}'..."):
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
    st.markdown("### 👤 Personalized Recommendations")
    st.markdown("Get movie recommendations based on your user profile and preferences.")
    
    if algo == "Content":
        st.info("ℹ️ For movie-based recommendations, switch to 'Movie Title' mode.")
        return

    # Add a more prominent user ID input
    with st.container():
        st.markdown("#### Your User Profile")
        user_id = st.number_input(
            "Enter your User ID:",
            min_value=1,
            value=user_id,
            step=1,
            help="Enter a number between 1 and 100,000"
        )
    
    if user_id not in set(ratings["userId"].values):
        st.info("🌟 Welcome! Since you're new, here are some popular movies to get you started.")
        recs = movies_pop.sort_values("count", ascending=False).head(top_n)[["movieId", "title"]].copy()
        recs["score"] = movies_pop.sort_values("count", ascending=False).head(top_n)["count"].values
    else:
        with st.spinner("Analyzing your preferences..."):
            recs = (
                recommend_user_hybrid(user_id, ratings, movies, cosine_sim, movieid_to_index, cf_model, top_n)
                if algo == "Hybrid"
                else recommend_user_cf(user_id, ratings, movies, cf_model, top_n)
            )
    
    st.markdown(f"#### Recommended for You (User #{user_id})")
    show_recs(recs)


def main():
    # Configure page
    st.set_page_config(
        page_title="🎬 MovieMatch - Smart Movie Recommendations",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #FF4B4B;
        }
        .stTextInput input {
            border-radius: 20px;
            padding: 0.5rem 1rem;
        }
        .stRadio > div {
            flex-direction: row;
            gap: 1rem;
        }
        .stRadio > label {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with description
    st.title("🎬 MovieMatch")
    st.markdown("### Discover your next favorite movie with AI-powered recommendations")
    
    # Load data
    with st.spinner("Loading movie database..."):
        ratings, movies, movies_content, cosine_sim, movieid_to_index, titles, movies_pop = load_data()
        cf_model = train_cf(ratings)
    
    # Sidebar controls
    with st.sidebar:
        st.image("https://img.icons8.com/3d-fluency/94/popcorn.png", width=80)
        st.header("Recommendation Settings")
        
        mode = st.radio(
            "Recommendation Type",
            ["🎥 Movie Title", "👤 User Profile"],
            index=0,
            format_func=lambda x: x.split(" ")[1:][0] if " " in x else x
        )
        
        algo = st.radio(
            "Algorithm",
            ["🎯 Content", "🔄 CF", "✨ Hybrid"],
            index=0,
            format_func=lambda x: x.split(" ")[1] if " " in x else x,
            help="Content: Based on movie similarity\nCF: Collaborative Filtering\nHybrid: Combines both approaches"
        )
        
        top_n = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="How many recommendations would you like to see?"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This recommender uses advanced algorithms to suggest movies 
        based on your preferences or similar titles.
        \n\n*Select 'Movie Title' to find similar movies.*  
        *Or choose 'User Profile' for personalized picks.*
        """)
        
        if cf_model is None:
            st.warning("⚠️ Collaborative Filtering requires additional packages. Using content-based recommendations instead.")
    
    # Main content
    if mode == "🎥 Movie Title":
        render_movie_mode(titles, movies_content, cosine_sim, movies_pop, top_n, algo.split(" ")[0])
    else:
        render_user_mode(
            1,  # Default user ID
            ratings,
            movies,
            cosine_sim,
            movieid_to_index,
            movies_pop,
            cf_model,
            top_n,
            algo.split(" ")[0],
        )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()


import numpy as np


def build_movieid_to_index(movies_content):
    return (
        movies_content.reset_index()
        .set_index("movieId")["index"]
        .to_dict()
    )


def normalize_scores(scores):
    if not scores:
        return {}

    values = np.array(list(scores.values()), dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if vmax == vmin:
        return dict.fromkeys(scores, 0.0)

    denom = vmax - vmin
    return {k: (float(v) - vmin) / denom for k, v in scores.items()}


def get_content_scores(
    user_id,
    ratings,
    movies_content,
    cosine_sim,
    movieid_to_index,
    liked_threshold=4.0,
):
    liked_movies = ratings[
        (ratings["userId"] == user_id) & (ratings["rating"] >= liked_threshold)
    ]["movieId"].unique()

    liked_indices = [
        movieid_to_index[mid]
        for mid in liked_movies
        if mid in movieid_to_index
    ]

    if len(liked_indices) == 0:
        return {}

    sim_scores = cosine_sim[liked_indices].mean(axis=0)
    return {
        movies_content.iloc[i]["movieId"]: float(sim_scores[i])
        for i in range(len(sim_scores))
    }


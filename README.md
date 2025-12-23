# movie-recommender

A small demo recommender-system project that showcases content-based, collaborative, and hybrid recommendation approaches with a simple Streamlit front-end.

## Contents

- `app/` — Streamlit demo UI and wiring (entry: `app/streamlit_app.py`).
- `src/` — Recommender implementations and helpers (`content.py`, `collaborative.py`, `hybrid.py`, `preprocessing.py`).
- `data/` — Raw and processed datasets (includes MovieLens `ml-100k` and `data/processed`).
- `notebooks/` — Reference notebooks with experiments and working code cells to port into `src/`.

## Quickstart

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

Optional: provide a TMDB API key in the app sidebar to display movie posters.

## What this project demonstrates

- Content-based recommendations using movie metadata and similarity measures.
- Collaborative filtering (user/item matrices and similarity or factorization).
- A simple hybrid approach that combines both signals for better coverage and personalization.

## Development notes

- Notebooks in `notebooks/` contain canonical implementations — when porting, move data loading to `src/preprocessing.py` and expose pure functions in `src/`.
- To add a new recommender, create `src/my_recommender.py` with a `recommend(items, k=10)`-style API and import it in `app/streamlit_app.py` for the demo.
- Expect datasets under `data/`. If a dataset is missing, add a loader that clearly instructs where to place data.



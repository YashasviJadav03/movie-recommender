<!-- .github/copilot-instructions.md
     Auto-generated guidance for AI coding agents working on this repo.
     If you already have guidance here, merge calmly: preserve non-conflicting
     sections and keep project-specific examples.
-->

# Copilot instructions for movie-recommender

Short actionable notes to help an AI coding assistant be productive in this
repository. Focus on discoverable patterns, important files, and concrete
examples rather than generic advice.

- **Big picture**: a small ML demo app with a Streamlit front end and
  recommender implementations in `src/`. Experimentation and reference
  notebooks live in `notebooks/` and datasets are expected in `data/`.

- **Key files / roles**:
  - `app/streamlit_app.py` — Streamlit entry point (UI + wiring). Agents
    should update this file to expose new recommender endpoints or demo
    controls.
  - `src/content.py` — content-based recommender code (feature extraction,
    similarity computations).
  - `src/collaborative.py` — collaborative filtering logic (user/item
    matrices, similarity, matrix factorization wrappers).
  - `src/hybrid.py` — glue code that composes content and collaborative
    approaches.
  - `src/preprocessing.py` — data cleaning and feature construction helpers.
  - `notebooks/` — canonical experiments and reference implementations. Use
    notebooks as the behavioral spec when the `src/` modules are incomplete.

- **Data layout**: Expect datasets under `data/`. Use relative paths from the
  repo root (e.g., `data/movies.csv`). If a dataset is missing, prefer
  adding a small loader that raises a clear error pointing to `data/`.

- **Runtime / developer workflows**:
  - Run the Streamlit demo locally:

    `streamlit run app/streamlit_app.py`

    (If `streamlit` isn't installed, create a venv and install dependencies.)
  - Open notebooks in `notebooks/` to reproduce experiments. Notebooks are
    typically the clearest source of working code for model training and
    evaluation.
  - There is no test suite or CI configuration in the repository; when
    implementing logic, add a small smoke test or script under `tests/` or
    a `scripts/` folder so maintainers can run it quickly.

- **Code & naming conventions (observed from repository structure)**:
  - One module per algorithmic approach: add new recommenders under `src/`.
  - Keep IO and model logic separate: prefer functions like `load_data()` in
    `preprocessing.py` and `recommend()` in algorithm modules.
  - Notebooks are treated as reference implementations; if you port notebook
    code into `src/`, maintain the same function names and signatures where
    practical.

- **Integration points**:
  - `app/streamlit_app.py` composes UI controls and imports from `src/`.
  - When adding a new feature, update the Streamlit app to import the new
    module and expose parameters via the UI.

- **Practical examples** (follow these when editing files):
  - Add a new recommender module:

    Create `src/my_new_recommender.py` with a `recommend(items, k=10)` API,
    then in `app/streamlit_app.py` add `from src.my_new_recommender import recommend`
    and wire it to a button/control in the UI.

  - Port notebook code into `src/`:

    1. Locate the working cell in `notebooks/XX_*.ipynb` that implements the
       algorithm.
    2. Move data loading into `src/preprocessing.py` as `load_data()`.
    3. Implement model logic as pure functions with explicit inputs and
       return values; avoid top-level side effects so unit tests can import
       the functions.

- **What to look for when a file is empty or missing**:
  - Notebooks often contain the canonical implementation. If `src/` files
    are empty (as they are now), prefer extracting working code from the
    notebooks and creating minimal, well-named functions in `src/`.

- **Limitations discovered**:
  - There is no `requirements.txt` or `pyproject.toml` in the repo. Before
    running the app, inspect notebooks for `import` statements to infer
    dependencies (common ones are `streamlit`, `pandas`, `numpy`, `scikit-learn`).

- **If you change project structure**:
  - Update this file with new import patterns and add a short example so
    future agents don't need to rediscover the layout.

If anything here is unclear or you want more examples (e.g., a small
reference implementation for a specific algorithm), tell me which area to
expand and I'll iterate.

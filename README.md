# Movie Recommendation System

A movie recommendation system implementing content-based filtering, collaborative filtering, and hybrid approaches using Python, scikit-learn, and Streamlit.

## Overview

This project demonstrates three recommendation strategies with an interactive web interface for exploring and comparing different recommendation methods.

## Features

- Content-based filtering using TF-IDF vectorization and cosine similarity
- Collaborative filtering using SVD matrix factorization
- Hybrid approach combining content-based and collaborative methods
- Interactive Streamlit web interface
- Fuzzy search for movie titles
- Pre-computed similarity matrices for efficient recommendations

## Dataset

This project uses the MovieLens 100K dataset containing 100,000 ratings from 943 users on 1,682 movies. The dataset is located in `data/ml-100k/` and includes movie metadata and user ratings.

Dataset source: http://grouplens.org/datasets/movielens/

## Project Structure

```
movie-recommender/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── content.py                # Content-based recommendation functions
│   ├── collaborative.py          # Collaborative filtering functions
│   ├── hybrid.py                 # Hybrid recommendation functions
│   └── preprocessing.py          # Data loading and preprocessing
├── tests/
│   └── test_recommendations.py   # Unit tests for recommendation functions
├── data/
│   ├── ml-100k/                  # MovieLens 100K raw dataset
│   └── processed/                # Generated processed data
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_Content_Based.ipynb    # Content-based development
│   ├── 03_Collaborative_Filtering.ipynb  # CF implementation
│   ├── 04_Hybrid_Model.ipynb     # Hybrid approach
│   └── 05_Evaluation.ipynb       # Model evaluation
├── config.py                     # Centralized configuration parameters
├── setup_data.py                 # Automated data preparation script
├── .gitignore                    # Git ignore rules
├── SUMMARY.md                    # Quick improvements summary
├── QUICK_START.md                # Quick reference guide
├── IMPROVEMENTS.md               # Detailed improvements documentation
├── CHANGELOG.md                  # Version history
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites
- Python 3.9 or higher

### Quick Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate processed data:
```bash
python setup_data.py
```

4. Run the application:
```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

For detailed commands and troubleshooting, see `QUICK_START.md`.

## Usage

The Streamlit interface provides two recommendation modes:

1. Movie Title Mode: Search for a movie and receive similar recommendations
2. User Profile Mode: Enter a user ID to receive personalized recommendations

Available algorithms:
- Content: Genre-based similarity
- CF: Collaborative filtering using user rating patterns
- Hybrid: Weighted combination of content and collaborative methods

Parameters:
- Number of recommendations (5-20)
- User ID for personalized recommendations

## Algorithms

### Content-Based Filtering
Uses TF-IDF vectorization and cosine similarity to recommend movies based on genre similarity. Implementation in `src/content.py`.

### Collaborative Filtering
Applies SVD matrix factorization to predict user preferences based on rating patterns. Implementation in `app/streamlit_app.py` using scikit-surprise.

### Hybrid Model
Combines content-based and collaborative filtering with weighted scoring. For movie-based recommendations, uses 70% content similarity and 30% popularity. For user-based recommendations, uses 60% collaborative filtering and 40% content similarity. Implementation in `src/hybrid.py`.

## Performance Metrics

The collaborative filtering model was evaluated on a test set (20% of the data) with the following results:

- RMSE: 0.9231
- MAE: 0.7282
- Precision@10: 0.6321
- Recall@10: 0.2825
- MAP@10: 0.8453

Model configuration: SVD with 50 factors, 30 epochs, learning rate 0.005, regularization 0.05.

## Development

### Running Tests
```bash
pytest tests/
```

### Data Processing
- Automated: Run `python setup_data.py` to generate all processed data
- Manual: Run notebooks in sequence (01-04) for exploratory analysis
- Processed data is saved to `data/processed/`

### Code Organization
- `src/content.py`: Content-based filtering functions
- `src/collaborative.py`: Collaborative filtering model training and prediction
- `src/hybrid.py`: Hybrid recommendation combining multiple approaches
- `src/preprocessing.py`: Data loading, cleaning, and feature engineering
- `app/streamlit_app.py`: Web interface and user interaction logic

## Dependencies

- streamlit: Web interface
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: TF-IDF and similarity metrics
- scikit-surprise: SVD collaborative filtering
- rapidfuzz: Fuzzy string matching
- matplotlib: Visualization in notebooks
- ipykernel: Jupyter notebook support
- pytest: Unit testing framework

Complete dependency list available in `requirements.txt`.

## Key Improvements

- Modular code architecture with separate modules for each algorithm
- Vectorized predictions for improved performance
- Input validation and error handling for robustness
- Consistent model parameters matching evaluation metrics
- Automated data preparation script for easy setup
- Unit tests for core recommendation functions
- Centralized configuration for easy parameter tuning
- Proper separation of concerns between UI and business logic
- Comprehensive documentation and quick start guide

For detailed information about improvements, see `IMPROVEMENTS.md` or `SUMMARY.md` for a quick overview.

## Notes

- Content-based recommendations are pre-computed for instant results
- Collaborative filtering model is trained and cached on first application load
- The cosine similarity matrix is pre-computed to optimize performance

## License

This project is for educational purposes.



import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src and root to path
root_path = Path(__file__).parents[1]
sys.path.insert(0, str(root_path / "src"))
sys.path.insert(0, str(root_path))

from content import get_content_recommendations, validate_movie_exists
from collaborative import validate_user_exists, get_user_recommendations, train_svd_model


class TestRecommendations(unittest.TestCase):
    """Unit tests for recommendation functions."""
    
    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests."""
        data_dir = Path(__file__).parents[1] / "notebooks" / "data" / "processed"
        
        if not data_dir.exists():
            data_dir = Path(__file__).parents[1] / "data" / "processed"
        
        if not (data_dir / "movies_content.csv").exists():
            raise FileNotFoundError(
                f"Test data not found. Run 'python setup_data.py' first.\n"
                f"Looked in: {data_dir}"
            )
        
        cls.movies_content = pd.read_csv(data_dir / "movies_content.csv")
        cls.movies_meta = pd.read_csv(data_dir / "movies_meta.csv")
        cls.ratings = pd.read_csv(data_dir / "ratings_clean.csv")
        cls.cosine_sim = np.load(data_dir / "content_cosine_sim.npy")
        
        # Train a small model for testing (faster)
        cls.cf_model = train_svd_model(cls.ratings.head(1000), n_factors=10, n_epochs=5)
    
    def test_content_recommendations_valid_movie(self):
        """Test content recommendations with a valid movie."""
        recs = get_content_recommendations(
            "Toy Story (1995)",
            self.movies_content,
            self.cosine_sim,
            top_n=5
        )
        
        self.assertEqual(len(recs), 5)
        self.assertIn("movieId", recs.columns)
        self.assertIn("title", recs.columns)
        self.assertIn("score", recs.columns)
        self.assertTrue(all(recs["score"] > 0))
    
    def test_content_recommendations_invalid_movie(self):
        """Test content recommendations with invalid movie."""
        recs = get_content_recommendations(
            "Nonexistent Movie (9999)",
            self.movies_content,
            self.cosine_sim,
            top_n=5
        )
        
        self.assertTrue(recs.empty)
    
    def test_validate_movie_exists(self):
        """Test movie validation."""
        self.assertTrue(validate_movie_exists("Toy Story (1995)", self.movies_content))
        self.assertFalse(validate_movie_exists("Fake Movie", self.movies_content))
    
    def test_validate_user_exists(self):
        """Test user validation."""
        self.assertTrue(validate_user_exists(1, self.ratings))
        self.assertFalse(validate_user_exists(99999, self.ratings))
    
    def test_recommendations_no_duplicates(self):
        """Test that recommendations don't include the input movie."""
        recs = get_content_recommendations(
            "Toy Story (1995)",
            self.movies_content,
            self.cosine_sim,
            top_n=10
        )
        
        self.assertNotIn("Toy Story (1995)", recs["title"].values)
    
    def test_score_ordering(self):
        """Test that recommendations are ordered by score."""
        recs = get_content_recommendations(
            "Toy Story (1995)",
            self.movies_content,
            self.cosine_sim,
            top_n=10
        )
        
        scores = recs["score"].values
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))
    
    def test_collaborative_recommendations(self):
        """Test collaborative filtering recommendations."""
        if self.cf_model is None:
            self.skipTest("Surprise library not available")
        
        recs = get_user_recommendations(
            1,
            self.ratings,
            self.movies_meta,
            self.cf_model,
            top_n=5
        )
        
        # Should return recommendations for valid user
        self.assertGreater(len(recs), 0)
        self.assertIn("score", recs.columns)
    
    def test_collaborative_invalid_user(self):
        """Test CF with invalid user returns empty."""
        if self.cf_model is None:
            self.skipTest("Surprise library not available")
        
        recs = get_user_recommendations(
            99999,
            self.ratings,
            self.movies_meta,
            self.cf_model,
            top_n=5
        )
        
        self.assertTrue(recs.empty)


if __name__ == "__main__":
    unittest.main()

# Tests

Unit tests for the movie recommendation system.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test
```bash
pytest tests/test_recommendations.py::TestRecommendations::test_content_recommendations_valid_movie
```

## Prerequisites

Tests require processed data. If you see errors about missing data files, run:
```bash
python setup_data.py
```

## Test Coverage

### Content-Based Recommendations
- Valid movie recommendations
- Invalid movie handling
- Movie existence validation
- No duplicate recommendations
- Score ordering

### Collaborative Filtering
- Valid user recommendations
- Invalid user handling
- User existence validation

### General
- Empty result handling
- Data structure validation
- Score range validation

## Adding New Tests

1. Add test methods to `test_recommendations.py`
2. Follow naming convention: `test_<feature>_<scenario>`
3. Use descriptive docstrings
4. Run tests to verify

Example:
```python
def test_new_feature(self):
    """Test description."""
    result = some_function()
    self.assertEqual(result, expected_value)
```

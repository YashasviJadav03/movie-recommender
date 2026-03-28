# Changelog

## Version 2.0 - Major Refactoring

### Fixed Issues

1. **Empty Source Files**
   - Implemented `src/preprocessing.py` with complete data preparation pipeline
   - Implemented `src/collaborative.py` with CF model training and prediction
   - Refactored `src/content.py` to provide modular functions
   - Enhanced `src/hybrid.py` with proper hybrid recommendation logic

2. **Code Organization**
   - Moved all recommendation logic from `streamlit_app.py` to `src/` modules
   - Created clear separation between UI and business logic
   - Removed redundant functions
   - Improved code reusability

3. **Model Inconsistency**
   - Fixed parameter mismatch between app (100 factors, 20 epochs) and evaluation (50 factors, 30 epochs)
   - Standardized to 50 factors, 30 epochs across all components
   - README metrics now accurately reflect app performance
   - Created `config.py` for centralized parameter management

4. **Error Handling**
   - Added `validate_user_exists()` function
   - Added `validate_movie_exists()` function
   - Functions return empty DataFrames instead of crashing
   - Added user-friendly error messages in UI
   - Fixed user ID range validation (1-943)

5. **Performance Issues**
   - Replaced `.apply()` with vectorized `predict_user_ratings()`
   - Optimized pandas operations
   - Improved numpy usage for numerical operations
   - Reduced redundant computations

6. **Missing Features**
   - Added unit tests in `tests/test_recommendations.py`
   - Created automated `setup_data.py` script
   - Added `.gitignore` for proper version control
   - Created `QUICK_START.md` for easy reference
   - Added `IMPROVEMENTS.md` documentation

7. **Data Dependency**
   - Created `setup_data.py` for one-command data preparation
   - Clear error messages when data is missing
   - Automatic directory creation
   - Fallback to notebooks still available

8. **Testing**
   - Added pytest to requirements
   - Created comprehensive unit tests
   - Tests for valid/invalid inputs
   - Tests for recommendation quality
   - Tests for score ordering

### New Files

- `config.py`: Centralized configuration
- `setup_data.py`: Automated data preparation
- `tests/test_recommendations.py`: Unit tests
- `.gitignore`: Version control configuration
- `QUICK_START.md`: Quick reference guide
- `IMPROVEMENTS.md`: Detailed improvement documentation
- `CHANGELOG.md`: This file

### Modified Files

- `app/streamlit_app.py`: Refactored to use modular functions
- `src/content.py`: Complete rewrite with proper functions
- `src/collaborative.py`: Implemented from scratch
- `src/hybrid.py`: Enhanced with config integration
- `src/preprocessing.py`: Implemented from scratch
- `requirements.txt`: Added pytest
- `README.md`: Updated with metrics and improvements

### Breaking Changes

None. The application interface remains the same.

### Migration Guide

If you have existing processed data:
1. No changes needed - the app will continue to work
2. Optionally run `python setup_data.py` to regenerate with new pipeline

If starting fresh:
1. Run `python setup_data.py` instead of notebooks
2. All functionality remains the same

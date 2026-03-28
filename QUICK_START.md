# Quick Start Guide

## First Time Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare data:
```bash
python setup_data.py
```

3. Run the app:
```bash
streamlit run app/streamlit_app.py
```

## Running Tests

```bash
pytest tests/
```

## Project Commands

### Data Preparation
```bash
# Automated (recommended)
python setup_data.py

# Manual (using preprocessing module)
python src/preprocessing.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_recommendations.py
```

### Application
```bash
# Start Streamlit app
streamlit run app/streamlit_app.py

# Start on specific port
streamlit run app/streamlit_app.py --server.port 8502
```

## Configuration

Edit `config.py` to adjust:
- SVD model parameters (factors, epochs, learning rate)
- Hybrid model weights
- Recommendation thresholds
- Search parameters

## Troubleshooting

### Missing Data Error
If you see "Missing processed data":
```bash
python setup_data.py
```

### Import Errors
Ensure you're in the project root and virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Test Failures
Make sure processed data exists before running tests:
```bash
python setup_data.py
pytest tests/
```

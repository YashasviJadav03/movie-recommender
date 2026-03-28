#!/usr/bin/env python
"""
Data preparation script for the movie recommendation system.
Run this script to generate all required processed data files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing import prepare_all_data


def main():
    print("=" * 60)
    print("Movie Recommender System - Data Setup")
    print("=" * 60)
    print()
    
    try:
        prepare_all_data()
        print()
        print("=" * 60)
        print("Setup complete! You can now run the Streamlit app:")
        print("  streamlit run app/streamlit_app.py")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure the MovieLens 100K dataset is in data/ml-100k/")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Executes the data collection pipeline."""
import sys
import pandas as pd
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection import get_youtube_comments, save_comments_to_csv
from src import config

def main():
    """Main function to run the data collection."""
    print("--- Starting Data Collection Pipeline ---")
    
    # Validate API key only when collecting data
    try:
        api_key = config.validate_api_key_for_collection()
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Ensure data directory exists
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RAW_DATA_DIR / config.RAW_COMMENTS_FILE
    
    # Collect comments with incremental saving
    comments = get_youtube_comments(
        api_key,  # Use validated API key
        config.VIDEO_ID, 
        output_path=output_path
    )
    
    if comments:
        # Final save (in case the last batch wasn't saved)
        save_comments_to_csv(comments, output_path)
        print(f"✅ Final save: {len(comments)} comments to {output_path}")
    else:
        print("❌ No comments collected. Check API key and video ID.")
        
    print("--- Data Collection Pipeline Finished ---")

if __name__ == "__main__":
    main()
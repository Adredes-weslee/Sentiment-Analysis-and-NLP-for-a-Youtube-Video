"""Executes the data preprocessing and VADER labeling pipeline."""
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_processing import clean_text, apply_vader_sentiment
from src import config

# Enable pandas progress bars
tqdm.pandas()

def main():
    """Main function to run the preprocessing."""
    print("--- Starting Preprocessing & Labeling Pipeline ---")
    
    # Load raw data
    raw_path = config.RAW_DATA_DIR / config.RAW_COMMENTS_FILE
    if not raw_path.exists():
        print(f"❌ Raw data file not found: {raw_path}")
        print("Run data collection first: python scripts/run_data_collection.py")
        return
        
    print("📁 Loading data...")
    df = pd.read_csv(raw_path)
    print(f"✅ Loaded {len(df)} raw comments.")
    
    # Debug: Show column names
    print(f"📋 CSV columns: {df.columns.tolist()}")
    
    # Auto-detect the comment column name
    comment_column = None
    possible_names = ['comment_raw', 'Comment', 'comment', 'text', 'comment_text']
    
    for col_name in possible_names:
        if col_name in df.columns:
            comment_column = col_name
            break
    
    if comment_column is None:
        print(f"❌ Could not find comment column. Available columns: {df.columns.tolist()}")
        print("Expected one of: comment_raw, Comment, comment, text, comment_text")
        return
    
    print(f"✅ Using comment column: '{comment_column}'")

    # Clean text data with progress bar - CORRECTED SYNTAX
    print("🧹 Cleaning text data...")
    df['comment_cleaned'] = df[comment_column].progress_apply(clean_text)
    
    # Apply sentiment analysis with progress bar - CORRECTED SYNTAX  
    print("🎯 Applying sentiment analysis...")
    df['sentiment'] = df['comment_cleaned'].progress_apply(apply_vader_sentiment)
    
    # Rename original column for consistency
    if comment_column != 'comment_raw':
        df = df.rename(columns={comment_column: 'comment_raw'})
    
    # Save processed data
    print("💾 Saving processed data...")
    processed_path = config.PROCESSED_DATA_DIR / config.PROCESSED_COMMENTS_FILE
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✅ Saved processed data to {processed_path}")
    
    # Show summary
    sentiment_counts = df['sentiment'].value_counts()
    print(f"📊 Sentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment}: {count:,} ({percentage:.1f}%)")
    
    print("--- Preprocessing Pipeline Finished ---")

if __name__ == "__main__":
    main()
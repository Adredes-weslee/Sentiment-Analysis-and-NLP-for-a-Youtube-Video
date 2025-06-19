"""Enhanced configuration for production deployment with automatic environment detection."""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# API Configuration - Properly loads from .env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "your_key_here":
    raise ValueError(
        "YouTube API key not found! Please:\n"
        "1. Copy .env.template to .env\n"
        "2. Add your actual YouTube API key to the .env file\n"
        "3. Get your key from: https://console.cloud.google.com/"
    )

VIDEO_ID = "kffacxfA7G4"  # Justin Bieber - Baby

# Data Paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_COMMENTS_FILE = "youtube_comments.csv"
PROCESSED_COMMENTS_FILE = "processed_comments.csv"

# Automatic Environment Detection
def is_streamlit_cloud():
    """Detect if running on Streamlit Community Cloud."""
    return (
        os.getenv("STREAMLIT_SHARING_MODE") is not None or  # Streamlit Cloud indicator
        os.getenv("HOSTNAME", "").startswith("streamlit-") or  # Streamlit Cloud hostname pattern
        "streamlit.app" in os.getenv("STREAMLIT_SERVER_ADDRESS", "")  # Streamlit Cloud domain
    )

# Model Configuration with Environment Detection
if is_streamlit_cloud():
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # 268MB - Fast startup
    HUGGING_FACE_MODEL = MODEL_NAME  # For backward compatibility
    print("🌐 Detected Streamlit Cloud - Using optimized model")
else:
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # 501MB - More accurate
    HUGGING_FACE_MODEL = MODEL_NAME  # For backward compatibility
    print("💻 Detected local environment - Using full model")

# Data Processing (from your original)
TEXT_MAX_LENGTH = 512  # For transformer models
BATCH_SIZE = 16  # For efficient inference
CONFIDENCE_THRESHOLD = 0.7  # For prediction confidence display

# Streamlit Configuration (enhanced)
APP_TITLE = "🎯 Advanced YouTube Sentiment Analyzer"
DASHBOARD_TITLE = "🎬 YouTube Sentiment Analysis Dashboard"
DASHBOARD_SUBTITLE = "Analyzing sentiment in Justin Bieber's 'Baby' comments"

APP_DESCRIPTION = """
This application demonstrates state-of-the-art sentiment analysis using transformer models.
Built from extensive research comparing multiple NLP approaches.
"""

# Alternative model options (for reference)
ALTERNATIVE_MODELS = {
    "multilingual": "nlptown/bert-base-multilingual-uncased-sentiment",
    "emotion": "j-hartmann/emotion-english-distilroberta-base",
    "fast": "distilbert-base-uncased-finetuned-sst-2-english",
    "accurate": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
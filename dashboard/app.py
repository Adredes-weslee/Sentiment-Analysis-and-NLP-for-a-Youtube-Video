"""Main Streamlit application file.

This is the entry point for the Streamlit dashboard. It sets up the main page
configuration and provides a brief introduction.
"""
import streamlit as st

st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="💬",
    layout="wide",
)

st.title("💬 YouTube Comment Sentiment Analysis Platform")
st.markdown("""
Welcome! This application uses a machine learning model to predict whether a
YouTube comment has a positive or negative sentiment.

**👈 Select 'Sentiment Classifier' from the sidebar to try it out!**

### How It Works
1.  **Data Collection**: Comments were scraped from a popular YouTube video.
2.  **Automated Labeling**: Each comment was automatically labeled as 'Positive' or 'Negative' using the VADER sentiment analysis tool.
3.  **Model Training**: A Logistic Regression classifier was trained on this labeled data to learn the patterns associated with each sentiment.

This platform allows you to use the trained model to get instant predictions on your own text.
""")



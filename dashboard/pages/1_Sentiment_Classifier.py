# Enhanced 1_Sentiment_Classifier.py
"""Advanced sentiment classification interface."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_processing import SentimentAnalyzer
from src.config import HUGGING_FACE_MODEL, CONFIDENCE_THRESHOLD

st.set_page_config(
    page_title="Sentiment Classifier", 
    page_icon="🎯",
    layout="wide"
)

# Initialize the analyzer
@st.cache_resource
def load_analyzer():
    """Load and cache the sentiment analyzer."""
    return SentimentAnalyzer(HUGGING_FACE_MODEL)

analyzer = load_analyzer()

# Main interface
st.title("🎯 Advanced Sentiment Analysis")
st.markdown("""
This tool uses state-of-the-art transformer models to analyze sentiment.
Enter text below to get detailed predictions with confidence scores.
""")

# Sidebar with model info
with st.sidebar:
    st.header("🔧 Model Information")
    st.info(f"**Model**: {HUGGING_FACE_MODEL}")
    st.info(f"**Confidence Threshold**: {CONFIDENCE_THRESHOLD}")
    
    st.header("📊 Usage Statistics")
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    st.metric("Predictions Made", st.session_state.prediction_count)

# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Text Input")
    
    # Sample texts for quick testing
    sample_texts = {
        "Positive Example": "I absolutely love this! The quality is amazing and exceeded my expectations.",
        "Negative Example": "This is terrible. Worst experience ever, completely disappointed.",
        "Neutral Example": "The weather today is cloudy with a chance of rain.",
        "Complex Example": "I don't hate it, but it's not great either. Mixed feelings about this."
    }
    
    selected_sample = st.selectbox("Quick Test Examples:", 
                                  ["Custom Input"] + list(sample_texts.keys()))
    
    if selected_sample != "Custom Input":
        default_text = sample_texts[selected_sample]
    else:
        default_text = ""
    
    user_input = st.text_area(
        "Enter your text here:",
        value=default_text,
        height=150,
        placeholder="Type or paste any text to analyze its sentiment..."
    )
    
    # Analysis options
    st.subheader("⚙️ Analysis Options")
    show_cleaned = st.checkbox("Show cleaned text", value=True)
    show_confidence = st.checkbox("Show confidence details", value=True)

with col2:
    st.subheader("📈 Quick Stats")
    
    if user_input:
        word_count = len(user_input.split())
        char_count = len(user_input)
        
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
        
        if word_count > 100:
            st.warning("⚠️ Long text detected. Consider shorter inputs for better accuracy.")

# Prediction button
if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            # Get prediction
            result = analyzer.predict(user_input)
            st.session_state.prediction_count += 1
            
            # Display results
            st.header("📊 Results")
            
            # Main prediction
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            if predicted_class.lower() == 'positive':
                st.success(f"✅ **{predicted_class}** sentiment detected!")
            elif predicted_class.lower() == 'negative':
                st.error(f"❌ **{predicted_class}** sentiment detected!")
            else:
                st.info(f"📊 **{predicted_class}** sentiment detected!")
            
            # Confidence visualization
            if show_confidence:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Confidence Scores")
                    
                    # Create DataFrame for scores
                    scores_df = pd.DataFrame([
                        {"Sentiment": k, "Confidence": v} 
                        for k, v in result['predictions'].items()
                    ])
                    
                    # Bar chart
                    fig = px.bar(
                        scores_df, 
                        x="Sentiment", 
                        y="Confidence",
                        color="Confidence",
                        color_continuous_scale="viridis",
                        title="Sentiment Confidence Scores"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Score Details")
                    
                    # Gauge chart for main prediction
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{predicted_class} Confidence"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed scores table
                st.subheader("📋 Detailed Scores")
                scores_display = pd.DataFrame([
                    {
                        "Sentiment": sentiment,
                        "Score": f"{score:.4f}",
                        "Percentage": f"{score*100:.2f}%"
                    }
                    for sentiment, score in result['predictions'].items()
                ]).sort_values("Score", ascending=False)
                
                st.dataframe(scores_display, use_container_width=True)
            
            # Show cleaned text if requested
            if show_cleaned:
                st.subheader("🧹 Preprocessed Text")
                st.code(result['cleaned_text'])
            
            # Save to history
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'text': user_input[:50] + "..." if len(user_input) > 50 else user_input,
                'prediction': predicted_class,
                'confidence': confidence
            })
    
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# History section
if 'prediction_history' in st.session_state and st.session_state.prediction_history:
    with st.expander("📚 Prediction History", expanded=False):
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
"""Overview of the research methodology and findings."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Research Overview", page_icon="🔬", layout="wide")


@st.cache_data
def load_committed_dataset():
    root = Path(__file__).resolve().parents[2]
    processed_path = root / "data" / "processed" / "processed_comments.csv"
    raw_path = root / "data" / "raw" / "youtube_comments.csv"
    for candidate in (processed_path, raw_path):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


dataset_df = load_committed_dataset()
current_total = len(dataset_df) if dataset_df is not None else None
current_distribution = None
if dataset_df is not None and "sentiment" in dataset_df.columns:
    current_distribution = (
        dataset_df["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )

st.title("🔬 Research Methodology & Findings")
st.markdown("""
This page summarizes the notebook-era modeling work behind the project.
The live dashboard uses the committed CSV corpus and a transformer-powered classifier, while the tables below preserve the earlier research comparisons as an archived experiment record.
""")

if current_total is not None:
    st.info(
        f"The committed dashboard dataset currently contains {current_total:,} comments. "
        "Use the Dataset Explorer for live corpus inspection; treat the figures on this page as research snapshots."
    )

# Methodology overview
st.subheader("📋 Research Methodology")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Collection & Processing:**
    - YouTube API collection from the target video plus replies
    - Notebook-era filtering and cleanup experiments
    - Duplicate removal and text normalization
    - Text cleaning & normalization
    - Emoji handling & URL removal
    - Committed CSVs provide the current dashboard corpus
    """)

with col2:
    st.markdown("""
    **Target Label Creation:**
    - VADER sentiment analysis for bootstrap labeling
    - Notebook comparisons across multiple pre-trained models
    - Transformer inference for the live classifier page
    - Ensemble experiments documented in the notebook
    - Binary positive/negative labels in the committed dashboard data
    """)

# Label Creation Performance
st.subheader("🎯 Target Label Creation Performance")
st.info("Performance on 100 hand-labeled comments for validation")

label_performance = pd.DataFrame({
    'Method': ['Hugging Face BART', 'Flair (IMDB)', 'VADER', 'Ensemble (Final)'],
    'Specificity': [0.65, 0.72, 0.68, 0.71],
    'NPV': [0.58, 0.65, 0.62, 0.65],
    'Harmonic_Mean': [0.61, 0.68, 0.65, 0.68]
})

fig_labels = px.bar(label_performance, x='Method', y=['Specificity', 'NPV', 'Harmonic_Mean'], 
                   title="Target Labeling Method Comparison",
                   barmode='group')
st.plotly_chart(fig_labels, use_container_width=True)

# Model Performance Comparison - ACTUAL RESULTS FROM NOTEBOOK
st.subheader("📊 Model Performance Comparison")
st.info("Results from extensive hyperparameter tuning and cross-validation")

# Real performance data from your notebook
performance_data = pd.DataFrame({
    'Model': [
        'Multinomial NB + Count Vec',
        'Multinomial NB + TF-IDF', 
        'Logistic Reg + Count Vec',
        'Logistic Reg + Word2Vec',
        'Random Forest + Count Vec',
        'Gradient Boosting + Count Vec',
        'Stacking + Count Vec',
        'Stacking + Word2Vec'
    ],
    'Cross_Val_F1': [0.60807, 0.56166, 0.64863, 0.61310, 0.60028, 0.58446, np.nan, np.nan],
    'Test_F1': [0.61526, 0.56727, 0.65556, 0.60733, 0.60019, 0.60653, 0.65228, 0.62219],
    'Train_F1': [0.65896, 0.61202, 0.73867, 0.62333, 0.70352, 0.88657, 0.79157, 0.73709],
    'Preprocessing': [
        'Lemmatization',
        'Lemmatization', 
        'Lemmatization',
        'None (Word2Vec)',
        'Lemmatization',
        'Lemmatization',
        'Lemmatization',
        'None (Word2Vec)'
    ]
})

# Performance visualization - FIXED VERSION
fig_perf = px.bar(performance_data, x='Model', y=['Test_F1', 'Cross_Val_F1'], 
                 title="Model Performance Comparison (F1-Score)",
                 barmode='group',
                 color_discrete_map={'Test_F1': '#1f77b4', 'Cross_Val_F1': '#ff7f0e'})

# FIXED: Use update_layout instead of update_xaxis
fig_perf.update_layout(
    xaxis_tickangle=-45,
    xaxis_title="Model",
    yaxis_title="F1-Score",
    legend_title="Metric Type"
)

st.plotly_chart(fig_perf, use_container_width=True)

# Detailed performance table
st.subheader("📋 Detailed Model Results")
st.dataframe(performance_data, use_container_width=True)

# Key findings
st.subheader("🎯 Key Research Findings")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **Main Conclusions:**
    1. **Logistic Regression + Count Vectorizer** achieved best performance (F1: 0.656)
    2. **Word2Vec embeddings** underperformed vs traditional vectorization
    3. **Class imbalance** (75/25 split) significantly impacted model performance
    4. **Maximum F1-score plateau** around 0.65 across all approaches
    """)

with col2:
    st.warning("""
    **Challenges Identified:**
    1. **Ambiguous comments** difficult even for human labelers
    2. **Majority positive sentiment** in "most disliked" video comments
    3. **Performance ceiling** suggests need for neutral class
    4. **Word embedding models** may need more preprocessing
    """)

# Hyperparameter insights
st.subheader("⚙️ Hyperparameter Optimization Insights")

with st.expander("🔧 Best Parameters Found"):
    st.markdown("""
    **Count Vectorizer + Logistic Regression (Best Model):**
    - Max Features: 15,000
    - N-gram Range: (1, 3)
    - Regularization (C): 0.25
    - Class Weight: Balanced for imbalance
    
    **Word2Vec Configuration:**
    - Vector Size: 200
    - Window Size: 5
    - Min Count: 10
    - Skip-gram architecture
    - 30 epochs training
    
    **Grid Search Scope:**
    - 720+ model combinations tested
    - 5-fold cross-validation
    - F1-score optimization for negative class
    """)

# Class imbalance impact
st.subheader("⚖️ Class Imbalance Analysis")

if current_distribution is not None and not current_distribution.empty:
    imbalance_data = current_distribution.copy()
    imbalance_data["Class"] = imbalance_data["Class"].str.title() + " Comments"
else:
    imbalance_data = pd.DataFrame({
        'Class': ['Positive Comments', 'Negative Comments'],
        'Count': [47408, 15628],
    })

fig_imbalance = px.pie(imbalance_data, values='Count', names='Class', 
                      title="Committed Dashboard Dataset Class Distribution")
st.plotly_chart(fig_imbalance, use_container_width=True)

# Future recommendations
st.subheader("🚀 Recommendations for Future Work")

recommendations = [
    "**Add Neutral Class**: Create 3-class system to handle ambiguous comments",
    "**Increase Dataset Size**: Collect from multiple videos for better generalization", 
    "**Advanced Preprocessing**: Apply stemming/lemmatization before Word2Vec",
    "**Transformer Models**: Experiment with BERT/RoBERTa for better performance",
    "**Active Learning**: Use human-in-the-loop for better label quality",
    "**Temporal Analysis**: Study sentiment changes over time"
]

for i, rec in enumerate(recommendations, 1):
    st.markdown(f"{i}. {rec}")

# Technical implementation details
with st.expander("💻 Technical Implementation Details"):
    st.markdown("""
    **Data Pipeline:**
    ```
    Raw Comments (99,941) → Language Filter → Deduplication → 
    Text Cleaning → Sentiment Labeling → Final Dataset (63,036)
    ```
    
    **Model Selection Process:**
    1. Baseline: Multinomial Naive Bayes
    2. Comparison: Traditional ML vs Word Embeddings
    3. Ensemble: Stacking multiple algorithms
    4. Evaluation: F1-score for minority class (negative)
    
    **Performance Constraints:**
    - Maximum F1-score: ~0.65 (performance ceiling)
    - Training time: 7.5 hours for Hugging Face labeling
    - Memory requirements: Sparse matrix optimization
    - Cross-validation: 5-fold stratified sampling
    """)

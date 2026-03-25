"""Main Streamlit entrypoint for the YouTube sentiment dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_home_stats():
    processed_path = PROJECT_ROOT / "data" / "processed" / "processed_comments.csv"
    raw_path = PROJECT_ROOT / "data" / "raw" / "youtube_comments.csv"

    if processed_path.exists():
        df = pd.read_csv(processed_path)
        sentiment_counts = (
            df["sentiment"].astype(str).str.strip().str.lower().value_counts()
            if "sentiment" in df.columns
            else pd.Series(dtype="int64")
        )
        return {
            "total_comments": len(df),
            "positive_comments": int(sentiment_counts.get("positive", 0)),
            "negative_comments": int(sentiment_counts.get("negative", 0)),
            "mode": "processed",
        }

    if raw_path.exists():
        df = pd.read_csv(raw_path)
        return {
            "total_comments": len(df),
            "positive_comments": None,
            "negative_comments": None,
            "mode": "raw",
        }

    return {
        "total_comments": None,
        "positive_comments": None,
        "negative_comments": None,
        "mode": "missing",
    }


def fmt_count(value):
    return f"{value:,}" if value is not None else "Pending"


st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .app-kicker {
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 0.82rem;
        font-weight: 700;
        color: #5c6176;
        margin-bottom: 0.35rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #4a5666;
        max-width: 48rem;
        margin-bottom: 1.2rem;
    }
    .home-callout {
        border: 1px solid rgba(82, 96, 153, 0.18);
        background: #f7f8fc;
        border-radius: 0.9rem;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0 1.1rem 0;
    }
    .home-callout strong {
        display: block;
        margin-bottom: 0.25rem;
        color: #1c2740;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

stats = load_home_stats()
st.sidebar.header("Start Here")
st.sidebar.info("Begin with Sentiment Classifier for a live prediction, then use Dataset Explorer for the committed corpus and Research Overview for the archived notebook findings.")
if stats["mode"] == "processed":
    st.sidebar.caption(
        f"Committed corpus loaded: {stats['total_comments']:,} processed comments."
    )
elif stats["mode"] == "raw":
    st.sidebar.caption(
        f"Raw corpus loaded: {stats['total_comments']:,} comments."
    )
else:
    st.sidebar.caption("No committed corpus detected.")

st.markdown(
    '<div class="app-kicker">NLP / audience reaction analysis</div>',
    unsafe_allow_html=True,
)
st.title("YouTube Comment Sentiment Analysis Platform")
st.markdown(
    '<div class="app-subtitle">A public demo that combines live comment scoring, committed corpus exploration, and archived modeling notes for the Justin Bieber “Baby” comment set.</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(3)
with metric_cols[0]:
    st.metric("Committed comments", fmt_count(stats["total_comments"]))
with metric_cols[1]:
    st.metric("Positive labels", fmt_count(stats["positive_comments"]))
with metric_cols[2]:
    st.metric("Negative labels", fmt_count(stats["negative_comments"]))

st.markdown(
    """
    <div class="home-callout">
      <strong>Use the dashboard in this order</strong>
      Start with <em>Sentiment Classifier</em> for the live transformer-backed prediction surface, then open <em>Dataset Explorer</em> to inspect the committed CSV corpus, and use <em>Research Overview</em> as the archive of the earlier notebook-era experiments.
    </div>
    """,
    unsafe_allow_html=True,
)

surface_cols = st.columns(3)
with surface_cols[0]:
    st.subheader("1. Score new comments")
    st.write(
        "Paste a comment and get a live sentiment prediction without needing the older notebook training pipeline."
    )
with surface_cols[1]:
    st.subheader("2. Inspect the committed corpus")
    st.write(
        "Review the processed dataset, sentiment distribution, and representative comments directly from the checked-in CSV artifacts."
    )
with surface_cols[2]:
    st.subheader("3. Read the archived research")
    st.write(
        "Use the research page as context for the original modeling work, not as the live source of truth for the current dashboard state."
    )

st.markdown("---")
st.subheader("How this demo is framed")
st.write(
    "The live dashboard is intentionally split into three surfaces because the project has both a runnable classifier and an older notebook-derived research record. The first screen now makes that distinction explicit instead of blending them together."
)

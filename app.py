import streamlit as st
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🎵 AI Music Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- BACKGROUND CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1507838153414-b4b713384a76?q=80&w=2940&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- AUDIO FEATURES ---
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness',
    'instrumentalness', 'liveness',
    'valence', 'tempo'
]

# --- MOOD PROFILES ---
MOOD_PROFILES = {
    "Happy": {
        'valence': 0.8,
        'energy': 0.7,
        'danceability': 0.7,
        'tempo': 120
    },
    "Sad": {
        'valence': 0.2,
        'energy': 0.3,
        'danceability': 0.3,
        'tempo': 70
    },
    "Chill": {
        'valence': 0.6,
        'energy': 0.4,
        'danceability': 0.5,
        'tempo': 90,
        'acousticness': 0.7
    },
    "Energetic": {
        'valence': 0.7,
        'energy': 0.9,
        'danceability': 0.8,
        'tempo': 130
    }
}

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["artists", "track_name"], inplace=True)
    df = df.drop_duplicates(subset=["artists", "track_name"])
    return df.reset_index(drop=True)

@st.cache_resource
def scale_features(df, features):
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[features])
    return scaled_matrix, scaler

# --- RECOMMENDATION FUNCTIONS ---
def recommend_by_track(track_name, df, scaled_matrix, n=5):
    if track_name not in df['track_name'].values:
        return None

    idx = df.index[df['track_name'] == track_name][0]
    similarity = cosine_similarity(
        scaled_matrix[idx].reshape(1, -1),
        scaled_matrix
    ).flatten()

    top_indices = similarity.argsort()[::-1]
    top_indices = [i for i in top_indices if i != idx][:n]

    return df.iloc[top_indices][["track_name", "artists"]]

def recommend_by_mood(mood, df, scaler, features, scaled_matrix, n=5):
    if mood not in MOOD_PROFILES:
        return None

    mood_vector = pd.DataFrame([[0.5] * len(features)], columns=features)

    for feature, value in MOOD_PROFILES[mood].items():
        if feature in mood_vector.columns:
            mood_vector[feature] = value

    mood_scaled = scaler.transform(mood_vector)
    similarity = cosine_similarity(mood_scaled, scaled_matrix).flatten()

    top_indices = similarity.argsort()[::-1][:n]
    return df.iloc[top_indices][["track_name", "artists"]]

def recommend_by_artist(artist, df, n=5):
    matches = df[df['artists'].str.contains(artist, case=False, na=False)]
    if matches.empty:
        return None
    return matches.sample(n=min(n, len(matches)))[["track_name", "artists"]]

def recommend_by_danceability(target, df, n=5):
    df_copy = df.copy()
    df_copy["diff"] = abs(df_copy["danceability"] - target)
    return df_copy.nsmallest(n, "diff")[["track_name", "artists", "danceability"]]

# --- MAIN APP ---
st.title("🎧 AI Music Recommender")

with st.spinner("Loading music dataset..."):
    df = load_data()
    scaled_matrix, scaler = scale_features(df, AUDIO_FEATURES)

st.header("Choose Recommendation Type")

mode = st.radio(
    "Recommendation method:",
    [
        "🎶 Based on track features",
        "😊 Based on mood",
        "🎤 Based on artist",
        "🕺 Based on danceability"
    ],
    horizontal=True
)

st.divider()

# --- UI MODES ---
if mode == "🎶 Based on track features":
    st.subheader("Find Similar Tracks")
    track = st.selectbox("🎵 Select a track", sorted(df["track_name"].unique()))
    n = st.slider("Number of recommendations", 1, 20, 5)

    if st.button("🚀 Recommend"):
        result = recommend_by_track(track, df, scaled_matrix, n)
        if result is not None:
            st.success(f"Tracks similar to **{track}**")
            st.dataframe(result, use_container_width=True)
        else:
            st.error("No recommendations found.")

elif mode == "😊 Based on mood":
    st.subheader("Mood-Based Music Discovery")
    mood = st.selectbox("Select mood", list(MOOD_PROFILES.keys()))
    n = st.slider("Number of recommendations", 1, 20, 5)

    if st.button("🚀 Recommend"):
        result = recommend_by_mood(mood, df, scaler, AUDIO_FEATURES, scaled_matrix, n)
        if result is not None:
            st.success(f"Songs for **{mood}** mood")
            st.dataframe(result, use_container_width=True)
        else:
            st.error("No recommendations found.")

elif mode == "🎤 Based on artist":
    st.subheader("Explore an Artist")
    artist = st.selectbox("Select artist", sorted(df["artists"].unique()))
    n = st.slider("Number of recommendations", 1, 20, 5)

    if st.button("🚀 Recommend"):
        result = recommend_by_artist(artist, df, n)
        if result is not None:
            st.success(f"More tracks by **{artist}**")
            st.dataframe(result, use_container_width=True)
        else:
            st.error("No tracks found.")

elif mode == "🕺 Based on danceability":
    st.subheader("Danceability-Based Recommendations")
    target = st.slider("Danceability (0.0 – 1.0)", 0.0, 1.0, 0.7, 0.01)
    n = st.slider("Number of recommendations", 1, 20, 5)

    if st.button("🚀 Recommend"):
        result = recommend_by_danceability(target, df, n)
        if result is not None:
            st.success("Recommended tracks")
            st.dataframe(result, use_container_width=True)
        else:
            st.error("No recommendations found.")

# --- SIDEBAR ---
st.sidebar.markdown("---")
with st.sidebar.expander("About this App"):
    st.markdown(
        """
        This AI-powered music recommender uses audio features and similarity
        metrics to suggest songs based on tracks, mood, artists, and danceability.
        """
    )
    st.markdown("**Developed by:** Your Name / Team")


import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="🎵 AI Music Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# FORCE SOLID BLUE BACKGROUND (KILLS ANY IMAGE CACHE)
# ----------------------------------------------------
st.markdown(
    """
    <style>
    /* Force remove any background image */
    .stApp {
        background-image: none !important;
        background: #0E4DA4 !important;
    }

    /* Ensure readable text */
    h1, h2, h3, h4, h5, h6,
    p, span, label, div {
        color: white !important;
    }

    /* Keep tables readable */
    .stDataFrame {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# CONSTANTS
# ----------------------------------------------------
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness',
    'instrumentalness', 'liveness',
    'valence', 'tempo'
]

MOOD_PROFILES = {
    "Happy": {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7, 'tempo': 120},
    "Sad": {'valence': 0.2, 'energy': 0.3, 'danceability': 0.3, 'tempo': 70},
    "Chill": {'valence': 0.6, 'energy': 0.4, 'danceability': 0.5, 'tempo': 90, 'acousticness': 0.7},
    "Energetic": {'valence': 0.7, 'energy': 0.9, 'danceability': 0.8, 'tempo': 130},
}

# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["track_name", "artists"], inplace=True)
    df.drop_duplicates(subset=["track_name", "artists"], inplace=True)
    return df.reset_index(drop=True)

@st.cache_resource
def scale_features(df, features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler

# ----------------------------------------------------
# RECOMMENDATION LOGIC
# ----------------------------------------------------
def recommend_by_track(track, df, scaled, n):
    idx = df.index[df["track_name"] == track][0]
    sim = cosine_similarity(scaled[idx].reshape(1, -1), scaled).flatten()
    top = sim.argsort()[::-1]
    top = [i for i in top if i != idx][:n]
    return df.iloc[top][["track_name", "artists"]]

def recommend_by_mood(mood, df, scaler, features, scaled, n):
    base = pd.DataFrame([[0.5] * len(features)], columns=features)
    for f, v in MOOD_PROFILES[mood].items():
        if f in base.columns:
            base[f] = v

    mood_vec = scaler.transform(base)
    sim = cosine_similarity(mood_vec, scaled).flatten()
    top = sim.argsort()[::-1][:n]
    return df.iloc[top][["track_name", "artists"]]

def recommend_by_artist(artist, df, n):
    matches = df[df["artists"].str.contains(artist, case=False, na=False)]
    return matches.sample(min(n, len(matches)))[["track_name", "artists"]]

def recommend_by_danceability(target, df, n):
    df = df.copy()
    df["diff"] = abs(df["danceability"] - target)
    return df.nsmallest(n, "diff")[["track_name", "artists", "danceability"]]

# ----------------------------------------------------
# MAIN APP
# ----------------------------------------------------
st.title("🎧 AI Music Recommender")

with st.spinner("Loading dataset..."):
    df = load_data()
    scaled, scaler = scale_features(df, AUDIO_FEATURES)

mode = st.radio(
    "Choose recommendation type",
    [
        "🎶 Based on track features",
        "😊 Based on mood",
        "🎤 Based on artist",
        "🕺 Based on danceability"
    ],
    horizontal=True
)

st.divider()

if mode == "🎶 Based on track features":
    track = st.selectbox("Select a track", sorted(df["track_name"].unique()))
    n = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend"):
        st.dataframe(recommend_by_track(track, df, scaled, n), use_container_width=True)

elif mode == "😊 Based on mood":
    mood = st.selectbox("Select mood", list(MOOD_PROFILES.keys()))
    n = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend"):
        st.dataframe(recommend_by_mood(mood, df, scaler, AUDIO_FEATURES, scaled, n), use_container_width=True)

elif mode == "🎤 Based on artist":
    artist = st.selectbox("Select artist", sorted(df["artists"].unique()))
    n = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend"):
        st.dataframe(recommend_by_artist(artist, df, n), use_container_width=True)

elif mode == "🕺 Based on danceability":
    target = st.slider("Danceability", 0.0, 1.0, 0.7, 0.01)
    n = st.slider("Number of recommendations", 1, 20, 5)
    if st.button("Recommend"):
        st.dataframe(recommend_by_danceability(target, df, n), use_container_width=True)


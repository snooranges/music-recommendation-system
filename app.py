import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page config for a wider layout and a nice title
st.set_page_config(page_title="🎵 Music Recommender", layout="wide", initial_sidebar_state="expanded")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["artists", "track_name"], inplace=True)
    df = df.drop_duplicates(subset=["artists", "track_name"])
    return df.reset_index(drop=True)

# Load the data
with st.spinner("Loading music data..."):
    df = load_data()
st.success("Music data loaded successfully!")


# --- Feature Scaling ---
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

@st.cache_resource
def scale_audio_features(data, features):
    scaler = StandardScaler()
    scaled_features_matrix = scaler.fit_transform(data[features])
    return scaled_features_matrix, scaler

# Scale audio features
with st.spinner("Processing audio features..."):
    scaled_features_matrix, audio_scaler = scale_audio_features(df, audio_features)
st.success("Audio features processed!")


# --- Recommendation Functions ---

def recommend_by_track(track_name, df, scaled_features_matrix, n=5):
    try:
        idx = df[df['track_name'] == track_name].index[0]
    except IndexError:
        return None
    
    similarity = cosine_similarity(scaled_features_matrix[idx].reshape(1, -1), scaled_features_matrix).flatten()
    
    top_indices = similarity.argsort()[-50:][::-1]
    top_indices = [i for i in top_indices if i != idx]
    
    if len(top_indices) == 0:
        return None
        
    selected = random.sample(list(top_indices), min(n, len(top_indices)))
    
    return df.iloc[selected][["track_name", "artists"]]

mood_profiles = {
    "Happy": {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7, 'tempo': 120},
    "Sad": {'valence': 0.2, 'energy': 0.3, 'danceability': 0.3, 'tempo': 70},
    "Chill": {'valence': 0.6, 'energy': 0.4, 'danceability': 0.5, 'tempo': 90, 'acousticness': 0.7},
    "Energetic": {'valence': 0.7, 'energy': 0.9, 'danceability': 0.8, 'tempo': 130},
}

def recommend_by_mood(mood, df, audio_scaler, audio_features, scaled_features_matrix, n=5):
    if mood not in mood_profiles:
        return None

    mood_vec_raw = pd.DataFrame([df[audio_features].mean().values], columns=audio_features)
    for feature, value in mood_profiles[mood].items():
        if feature in audio_features:
            mood_vec_raw[feature] = value

    mood_vec_scaled = audio_scaler.transform(mood_vec_raw[audio_features])
    
    similarity = cosine_similarity(mood_vec_scaled, scaled_features_matrix).flatten()
    
    top_indices = similarity.argsort()[-50:][::-1]
    
    selected = random.sample(list(top_indices), min(n, len(top_indices)))
    
    return df.iloc[selected][["track_name", "artists"]]

def recommend_by_artist(artist_name, df, n=5):
    artist_songs = df[df['artists'] == artist_name]
    
    if artist_songs.empty:
        return None
    
    if len(artist_songs) <= n:
        return artist_songs[["track_name", "artists"]].sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        return artist_songs[["track_name", "artists"]].sample(n=n, random_state=42).reset_index(drop=True)

def recommend_by_danceability(target_danceability, df, n=5):
    df['danceability_diff'] = abs(df['danceability'] - target_danceability)
    
    recommended_songs = df.nsmallest(n, 'danceability_diff')
    
    recommended_songs = recommended_songs.drop(columns=['danceability_diff'])
    
    return recommended_songs[["track_name", "artists", "danceability"]]


# --- Streamlit UI ---

st.title("🎧 Music Recommendation System")

# About section using an expander with detailed description
with st.expander("About this App"):
    st.markdown(
        """
        This application recommends music based on various criteria using a comprehensive dataset of track features.
        
        **Techniques Used:**
        - **Data Handling:** Utilizes `pandas` for efficient loading, cleaning, and manipulation of the music dataset.
        - **Feature Scaling:** Employs `StandardScaler` from `scikit-learn` to normalize numerical audio features, ensuring that all features contribute equally to similarity calculations.
        - **Content-Based Filtering:** Implements content-based recommendation logic. For track and mood-based recommendations, `cosine_similarity` from `scikit-learn` is used to find songs that are similar in their audio characteristics.
        - **User Interface:** Built entirely with `Streamlit`, allowing for an interactive and user-friendly web application with minimal code.
        
        **Explore recommendations based on:**
        - **Track Features**: Find songs similar to a selected track based on its audio characteristics.
        - **Mood**: Get songs that align with a chosen mood (e.g., Happy, Sad, Energetic).
        - **Artist**: Discover other songs by your favorite artist.
        - **Danceability**: Find songs suitable for dancing based on a desired 'danceability' score (0.0 to 1.0).
        """
    )
    st.write("---") 
    st.markdown("**Developed by:**")
    st.markdown("- Praneeth (2453-207-33085)")
    st.markdown("- Rishi (245322733124)")
    st.markdown("- Kevin (245322733127)")
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and scikit-learn.")


# Use a sidebar for main navigation/mode selection
st.sidebar.header("Recommendation Options")
mode = st.sidebar.radio(
    "Choose recommendation type:", 
    ["🎶 Based on track features", "😊 Based on mood only", "🎤 Based on artist", "🕺 Based on danceability"]
)

# Main content area based on selected mode
st.header(f"--- {mode} ---")

if mode == "🎶 Based on track features":
    st.subheader("Find Similar Tracks by Audio Features")
    selected_track = st.selectbox("🎵 Pick a track:", sorted(df["track_name"].unique()))
    num_recommendations = st.slider("🔢 How many recommendations?", 1, 20, 5)
    
    if st.button("🚀 Recommend by Track Features"):
        with st.spinner("Finding similar tracks..."):
            result = recommend_by_track(selected_track, df, scaled_features_matrix, n=num_recommendations)
        if selected_track:
            if result is not None and not result.empty:
                st.success(f"✨ Here are tracks similar to: **{selected_track}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("💡 Could not find similar tracks for the selected track. Try another one!")
        else:
            st.warning("⚠️ Please select a track to get recommendations.")

elif mode == "😊 Based on mood only":
    st.subheader("Discover Tracks for Your Mood")
    mood_choice = st.selectbox("😊 Select your mood:", list(mood_profiles.keys()))
    num_recommendations_mood = st.slider("🔢 How many recommendations?", 1, 20, 5)
    
    if st.button("🚀 Recommend by Mood"):
        with st.spinner(f"Finding tracks for {mood_choice} mood..."):
            result = recommend_by_mood(mood_choice, df, audio_scaler, audio_features, scaled_features_matrix, n=num_recommendations_mood)
        if mood_choice:
            if result is not None and not result.empty:
                st.success(f"🎉 Here are tracks recommended for mood: **{mood_choice}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("💡 Could not find tracks matching this mood profile. Try another mood!")
        else:
            st.warning("⚠️ Please select a mood.")

elif mode == "🎤 Based on artist":
    st.subheader("Explore More from an Artist")
    selected_artist = st.selectbox("🎤 Pick an artist:", sorted(df["artists"].unique()))
    num_recommendations_artist = st.slider("🔢 How many recommendations?", 1, 20, 5)
    
    if st.button("🚀 Recommend by Artist"):
        with st.spinner(f"Finding tracks by {selected_artist}..."):
            result = recommend_by_artist(selected_artist, df, n=num_recommendations_artist)
        if selected_artist:
            if result is not None and not result.empty:
                st.success(f"🎶 Here are other tracks by: **{selected_artist}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("💡 Artist not found or no other tracks available. Try another artist!")
        else:
            st.warning("⚠️ Please select an artist.")

elif mode == "🕺 Based on danceability":
    st.subheader("Find Tracks by Danceability Score")
    desired_danceability = st.slider("💃 Select desired danceability (0.0 = low, 1.0 = high):", 0.0, 1.0, 0.7, 0.05)
    num_recommendations_danceability = st.slider("🔢 How many recommendations?", 1, 20, 5)

    if st.button("🚀 Recommend by Danceability"):
        with st.spinner(f"Finding tracks with danceability around {desired_danceability:.2f}..."):
            result = recommend_by_danceability(desired_danceability, df, n=num_recommendations_danceability)
        if desired_danceability is not None:
            if result is not None and not result.empty:
                st.success(f"🕺🎶 Tracks with danceability close to: **{desired_danceability:.2f}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("💡 Could not find tracks for the selected danceability. Adjust the slider!")
        else:
            st.warning("⚠️ Please select a danceability value.")


# Dataset sample in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Sample (First 5 Rows)")
st.sidebar.dataframe(df.head(), height=200)

st.markdown("---")
st.info("🚀 Your music journey starts here!")

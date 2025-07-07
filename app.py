import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["artists", "track_name"], inplace=True)
    df = df.drop_duplicates(subset=["artists", "track_name"])
    return df.reset_index(drop=True)

df = load_data()

# Define numerical audio features for scaling and similarity calculation
audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

# Scale audio features
@st.cache_resource
def scale_audio_features(data, features):
    scaler = StandardScaler()
    scaled_features_matrix = scaler.fit_transform(data[features])
    return scaled_features_matrix, scaler

scaled_features_matrix, audio_scaler = scale_audio_features(df, audio_features)

# Recommend based on song (now track name)
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

# Mood keywords and their corresponding target audio feature values (example values)
mood_profiles = {
    "Happy": {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7, 'tempo': 120},
    "Sad": {'valence': 0.2, 'energy': 0.3, 'danceability': 0.3, 'tempo': 70},
    "Chill": {'valence': 0.6, 'energy': 0.4, 'danceability': 0.5, 'tempo': 90, 'acousticness': 0.7},
    "Energetic": {'valence': 0.7, 'energy': 0.9, 'danceability': 0.8, 'tempo': 130},
}

# Recommend based on mood using audio features
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

# Recommend based on artist
def recommend_by_artist(artist_name, df, n=5):
    artist_songs = df[df['artists'] == artist_name]
    
    if artist_songs.empty:
        return None
    
    if len(artist_songs) <= n:
        return artist_songs[["track_name", "artists"]].sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        return artist_songs[["track_name", "artists"]].sample(n=n, random_state=42).reset_index(drop=True)

# Recommend based on danceability
def recommend_by_danceability(target_danceability, df, n=5):
    # Calculate the absolute difference from the target danceability for all songs
    df['danceability_diff'] = abs(df['danceability'] - target_danceability)
    
    # Sort by this difference to find the closest songs
    # Use .nsmallest to efficiently get the top N smallest differences
    recommended_songs = df.nsmallest(n, 'danceability_diff')
    
    # Drop the temporary difference column
    recommended_songs = recommended_songs.drop(columns=['danceability_diff'])
    
    return recommended_songs[["track_name", "artists", "danceability"]]

# Streamlit UI
st.set_page_config(page_title="🎵 Music Recommender", layout="wide")
st.title("🎧 Music Recommendation System")

mode = st.radio(
    "Choose recommendation type:", 
    ["🎶 Based on track features", "😊 Based on mood only", "🎤 Based on artist", "🕺 Based on danceability"]
)

if mode == "🎶 Based on track features":
    st.header("Similar Tracks by Audio Features")
    selected_track = st.selectbox("Pick a track:", sorted(df["track_name"].unique()))
    num_recommendations = st.slider("How many recommendations?", 1, 20, 5)
    
    if st.button("🎯 Recommend by Track Features"):
        if selected_track:
            result = recommend_by_track(selected_track, df, scaled_features_matrix, n=num_recommendations)
            if result is not None and not result.empty:
                st.success(f"Tracks similar to: **{selected_track}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("Could not find similar tracks for the selected track.")
        else:
            st.warning("Please select a track.")

elif mode == "😊 Based on mood only":
    st.header("Tracks Recommended for Mood")
    mood_choice = st.selectbox("Select your mood:", list(mood_profiles.keys()))
    num_recommendations_mood = st.slider("How many recommendations?", 1, 20, 5)
    
    if st.button("🎯 Recommend by Mood"):
        if mood_choice:
            result = recommend_by_mood(mood_choice, df, audio_scaler, audio_features, scaled_features_matrix, n=num_recommendations_mood)
            if result is not None and not result.empty:
                st.success(f"🎧 Tracks recommended for mood: **{mood_choice}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("Could not find tracks matching this mood profile.")
        else:
            st.warning("Please select a mood.")

elif mode == "🎤 Based on artist":
    st.header("Other Tracks by Artist")
    selected_artist = st.selectbox("Pick an artist:", sorted(df["artists"].unique()))
    num_recommendations_artist = st.slider("How many recommendations?", 1, 20, 5)
    
    if st.button("🎯 Recommend by Artist"):
        if selected_artist:
            result = recommend_by_artist(selected_artist, df, n=num_recommendations_artist)
            if result is not None and not result.empty:
                st.success(f"Other tracks by: **{selected_artist}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("Artist not found or no other tracks available.")
        else:
            st.warning("Please select an artist.")

elif mode == "🕺 Based on danceability":
    st.header("Tracks Recommended by Danceability")
    # Danceability is typically from 0.0 to 1.0
    desired_danceability = st.slider("Select desired danceability:", 0.0, 1.0, 0.7, 0.05)
    num_recommendations_danceability = st.slider("How many recommendations?", 1, 20, 5)

    if st.button("🎯 Recommend by Danceability"):
        if desired_danceability is not None:
            result = recommend_by_danceability(desired_danceability, df, n=num_recommendations_danceability)
            if result is not None and not result.empty:
                st.success(f"🎶 Tracks with danceability close to: **{desired_danceability:.2f}**")
                st.dataframe(result.style.set_properties(**{'text-align': 'left'}))
            else:
                st.info("Could not find tracks for the selected danceability.")
        else:
            st.warning("Please select a danceability value.")

st.sidebar.header("Dataset Sample")
st.sidebar.dataframe(df.head())
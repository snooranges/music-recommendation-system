import streamlit as st
import pandas as pd
import random
import threading
import av
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- PAGE CONFIG ---
st.set_page_config(page_title="🎵 AI Music Recommender", layout="wide", initial_sidebar_state="collapsed")

# --- CSS for Background ---
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

# --- CONSTANTS & MAPPINGS ---
AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

MOOD_PROFILES = {
    "Happy": {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7, 'tempo': 120},
    "Sad": {'valence': 0.2, 'energy': 0.3, 'danceability': 0.3, 'tempo': 70},
    "Chill": {'valence': 0.6, 'energy': 0.4, 'danceability': 0.5, 'tempo': 90, 'acousticness': 0.7},
    "Energetic": {'valence': 0.7, 'energy': 0.9, 'danceability': 0.8, 'tempo': 130},
}

EMOTION_TO_MOOD_MAPPING = {
    'happy': 'Happy',
    'sad': 'Sad',
    'neutral': 'Chill',
    'surprise': 'Energetic',
    'angry': 'Energetic',
    'fear': 'Sad',
    'disgust': 'Chill'
}

# --- DATA LOADING & PROCESSING ---
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.dropna(subset=["artists", "track_name"], inplace=True)
    df = df.drop_duplicates(subset=["artists", "track_name"])
    return df.reset_index(drop=True)

@st.cache_resource
def scale_audio_features(data, features):
    scaler = StandardScaler()
    scaled_features_matrix = scaler.fit_transform(data[features])
    return scaled_features_matrix, scaler

# --- EMOTION RECOGNITION SETUP ---
lock = threading.Lock()
emotion_data = {"dominant_emotion": "neutral"}

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list) and len(analysis) > 0:
                dominant_emotion = analysis[0]['dominant_emotion']
                with lock:
                    emotion_data["dominant_emotion"] = dominant_emotion
                cv2.putText(img, f"Emotion: {dominant_emotion.capitalize()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- RECOMMENDATION LOGIC ---
def recommend_by_track(track_name, df, scaled_features_matrix, n=5):
    try:
        idx = df[df['track_name'] == track_name].index[0]
    except IndexError:
        return None
    
    similarity = cosine_similarity(scaled_features_matrix[idx].reshape(1, -1), scaled_features_matrix).flatten()
    top_indices = [i for i in similarity.argsort()[-50:][::-1] if i != idx]
    
    if not top_indices: return None
    selected = random.sample(top_indices, min(n, len(top_indices)))
    return df.iloc[selected][["track_name", "artists"]]

def recommend_by_mood(mood, df, audio_scaler, features, scaled_features_matrix, n=5):
    if mood not in MOOD_PROFILES: return None

    mood_vec_raw = pd.DataFrame([df[features].mean().values], columns=features)
    for feature, value in MOOD_PROFILES[mood].items():
        if feature in features:
            mood_vec_raw[feature] = value

    mood_vec_scaled = audio_scaler.transform(mood_vec_raw[features])
    similarity = cosine_similarity(mood_vec_scaled, scaled_features_matrix).flatten()
    
    top_indices = similarity.argsort()[-50:][::-1]
    selected = random.sample(list(top_indices), min(n, len(top_indices)))
    return df.iloc[selected][["track_name", "artists"]]

def recommend_by_artist(artist_name, df, n=5):
    artist_songs = df[df['artists'] == artist_name]
    if artist_songs.empty: return None
    
    sample_n = min(n, len(artist_songs))
    return artist_songs.sample(n=sample_n)[["track_name", "artists"]]

def recommend_by_danceability(target_danceability, df, n=5):
    df_copy = df.copy()
    df_copy['danceability_diff'] = abs(df_copy['danceability'] - target_danceability)
    recommended_songs = df_copy.nsmallest(n, 'danceability_diff')
    return recommended_songs[["track_name", "artists", "danceability"]]

# --- MAIN APP UI ---
st.title("🎧 AI Music Recommender")

with st.spinner("Loading music data..."):
    df = load_data()
    scaled_features_matrix, audio_scaler = scale_audio_features(df, AUDIO_FEATURES)

st.header("Choose Recommendation Type:")
mode = st.radio(
    "Select your preferred method:",
    ["🎶 Based on track features", "😊 Based on mood only", "🎤 Based on artist", "🕺 Based on danceability", "🙂 Based on your emotion"],
    horizontal=True
)
st.write("---")

if mode == "🎶 Based on track features":
    st.subheader("Find Similar Tracks by Audio Features")
    selected_track = st.selectbox("🎵 Pick a track:", sorted(df["track_name"].unique()))
    num_recs = st.slider("🔢 How many recommendations?", 1, 20, 5, key="track_slider")
    if st.button("🚀 Recommend by Track"):
        result = recommend_by_track(selected_track, df, scaled_features_matrix, n=num_recs)
        if result is not None:
            st.success(f"✨ Here are tracks similar to: **{selected_track}**")
            st.dataframe(result)
        else:
            st.error("Could not find recommendations.")

elif mode == "😊 Based on mood only":
    st.subheader("Discover Tracks for Your Mood")
    mood_choice = st.selectbox("😊 Select your mood:", list(MOOD_PROFILES.keys()))
    num_recs = st.slider("🔢 How many recommendations?", 1, 20, 5, key="mood_slider")
    if st.button("🚀 Recommend by Mood"):
        result = recommend_by_mood(mood_choice, df, audio_scaler, AUDIO_FEATURES, scaled_features_matrix, n=num_recs)
        if result is not None:
            st.success(f"🎉 Here are tracks for a **{mood_choice}** mood:")
            st.dataframe(result)
        else:
            st.error("Could not find recommendations.")

elif mode == "🎤 Based on artist":
    st.subheader("Explore More from an Artist")
    selected_artist = st.selectbox("🎤 Pick an artist:", sorted(df["artists"].unique()))
    num_recs = st.slider("🔢 How many recommendations?", 1, 20, 5, key="artist_slider")
    if st.button("🚀 Recommend by Artist"):
        result = recommend_by_artist(selected_artist, df, n=num_recs)
        if result is not None:
            st.success(f"🎶 Here are other tracks by: **{selected_artist}**")
            st.dataframe(result)
        else:
            st.error("Could not find recommendations.")

elif mode == "🕺 Based on danceability":
    st.subheader("Find Tracks by Danceability Score")
    desired_danceability = st.slider("💃 Select desired danceability (0.0 = low, 1.0 = high):", 0.0, 1.0, 0.7, 0.01)
    num_recs = st.slider("🔢 How many recommendations?", 1, 20, 5, key="dance_slider")
    if st.button("🚀 Recommend by Danceability"):
        result = recommend_by_danceability(desired_danceability, df, n=num_recs)
        if result is not None:
            st.success(f"🕺 Tracks with danceability close to: **{desired_danceability:.2f}**")
            st.dataframe(result)
        else:
            st.error("Could not find recommendations.")
            
elif mode == "🙂 Based on your emotion":
    st.subheader("Get Recommendations Based on Your Live Emotion")
    st.write("Enable your webcam below. The app will detect your emotion in real-time. Then, click the button!")
    webrtc_streamer(key="emotion-detection", video_processor_factory=EmotionProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    num_recs = st.slider("🔢 How many recommendations?", 1, 20, 5, key="emotion_slider")
    if st.button("🚀 Recommend Songs for My Emotion"):
        with lock:
            detected_emotion = emotion_data["dominant_emotion"]
        
        st.info(f"Detected emotion: **{detected_emotion.capitalize()}**")
        target_mood = EMOTION_TO_MOOD_MAPPING.get(detected_emotion, "Chill")
        st.write(f"Finding songs for a '{target_mood}' mood...")
        
        result = recommend_by_mood(target_mood, df, audio_scaler, AUDIO_FEATURES, scaled_features_matrix, n=num_recs)
        if result is not None:
            st.success(f"🎉 Here are songs recommended for your '{target_mood}' mood:")
            st.dataframe(result)
        else:
            st.error("Could not find recommendations.")

# --- ABOUT SECTION ---
st.sidebar.markdown("---")
with st.sidebar.expander("About this App"):
    st.markdown(
        """
        This AI-powered application recommends music using several advanced techniques. 
        It analyzes audio features and can even detect your live emotion via webcam to suggest a playlist.
        """
    )
    st.markdown("**Developed by:** Your Name/Group")

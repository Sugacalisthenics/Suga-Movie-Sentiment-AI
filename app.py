import streamlit as st
from transformers import pipeline
import pandas as pd
import zipfile
import os

# 1. Naya Super-Smart AI load karna (Double-negation support)
@st.cache_resource
def load_ai_model():
    # Yeh model tricky English aur double negatives bahut achhe se samajhta hai
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

analyzer = load_ai_model()

# 2. Load Full Data (Netflix titles)
@st.cache_data
def load_full_data():
    zip_path = 'netflix_titles.csv.zip'
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            target_file = 'netflix_movies2025.csv'
            if target_file in z.namelist():
                with z.open(target_file) as f:
                    return pd.read_csv(f)
    return pd.DataFrame()

df = load_full_data()

# 3. UI Design
st.set_page_config(page_title="Suga's Movie Insights", layout="wide")

st.title("🍿 Suga's Movie Sentiment AI & Explorer")

# Sidebar for Movie Details
st.sidebar.title("🎬 Movie Details")
if not df.empty:
    movie_list = df['title'].sort_values().unique()
    selected_movie = st.selectbox("Select a Movie:", movie_list)
    
    # Filter data for selected movie
    movie_info = df[df['title'] == selected_movie].iloc[0]
    
    # Display Details in Sidebar
    st.sidebar.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=300", use_container_width=True)
    st.sidebar.write(f"**Year:** {movie_info.get('release_year', 'N/A')}")
    st.sidebar.write(f"**Director:** {movie_info.get('director', 'N/A')}")
    st.sidebar.info(f"**Plot:** {movie_info.get('description', 'N/A')}")
else:
    st.sidebar.error("Data load nahi ho paya!")
    selected_movie = "Manual Review"

# Main Section for Sentiment Analysis
st.subheader(f"Analyze your thoughts on: {selected_movie}")
user_input = st.text_area("Review yahan likhein:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        with st.spinner('Suga AI is analyzing your review...'):
            # Naye AI se inference
            result = analyzer(user_input)[0]
            
            # Label ko text mein badal kar UPPERCASE kar liya taaki exact match ho
            label = str(result['label']).upper() 
            confidence = round(result['score'] * 100, 2)
            
            # Simple POSITIVE/NEGATIVE check
            if label == 'POSITIVE':
                st.success(f"POSITIVE! 😊 Aapko **{selected_movie}** pasand aayi.")
                st.balloons()
            else:
                st.error(f"NEGATIVE! 😞 Aapko **{selected_movie}** achhi nahi lagi.")
                
            st.caption(f"AI Confidence: {confidence}% | Sentiment Code: {label}")
    else:
        st.warning("Pehle kuch review toh likhiye!")

st.markdown("---")
st.caption("Model: DistilBERT SST-2 | Smart Reasoning AI")

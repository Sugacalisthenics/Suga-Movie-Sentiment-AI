import streamlit as st
import pickle
import pandas as pd
import re
import nltk
import zipfile
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 1. Setup
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

# 2. Load ML Components
model = pickle.load(open('movie_sentiment_model.pkl', 'rb'))
cv = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# 3. Load Full Data (Not just titles)
@st.cache_data
def load_full_data():
    zip_path = 'netflix_titles.csv.zip'
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            target_file = 'netflix_movies2025.csv' # As seen in your zip folder
            if target_file in z.namelist():
                with z.open(target_file) as f:
                    return pd.read_csv(f)
    return pd.DataFrame()

df = load_full_data()

# 4. Logic Functions
def clean_review(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    return ' '.join(review)

# 5. UI Design
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
    st.sidebar.write(f"**Year:** {movie_info['release_year']}")
    st.sidebar.write(f"**Director:** {movie_info.get('director', 'N/A')}")
    st.sidebar.info(f"**Plot:** {movie_info['description']}")
else:
    st.sidebar.error("Data load nahi ho paya!")
    selected_movie = "Manual Review"

# Main Section for Sentiment Analysis
st.subheader(f"Analyze your thoughts on: {selected_movie}")
user_input = st.text_area("Review yahan likhein:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_review(user_input)
        vector = cv.transform([cleaned])
        prediction = model.predict(vector)
        
        if prediction[0] == 1:
            st.success(f"POSITIVE! 😊 Aapko **{selected_movie}** pasand aayi.")
            st.balloons()
        else:
            st.error(f"NEGATIVE! 😞 Aapko **{selected_movie}** achhi nahi lagi.")
    else:
        st.warning("Pehle kuch review toh likhiye!")

st.markdown("---")
st.caption("Model Accuracy: 85.24% | Trained on 50k IMDB Reviews")
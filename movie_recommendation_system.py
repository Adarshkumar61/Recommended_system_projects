import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st # not installed in this env.
from sklearn.impute import SimpleImputer

def main():
    # Load data
    data = pd.read_csv(r'C:\Users\adars\Adarsh\movies.csv')

    # Combine movie data
    movie_data = ['genres', 'keywords', 'tagline', 'cast', 'director']
    simple_imputer = SimpleImputer(strategy='constant', fill_value='')
    data[movie_data] = simple_imputer.fit_transform(data[movie_data])

    # Ensure 'index' column exists
    if 'index' not in data.columns:
        data['index'] = data.index

    combine_data = data['genres'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast'] + ' ' + data['director']

    # Create TF-IDF converter
    converter = TfidfVectorizer()
    feature = converter.fit_transform(combine_data)

    # Calculate cosine similarity
    similarity = cosine_similarity(feature)

    st.title("Movie Recommendation System")

    # User input
    title_of_movie = difflib.get_close_matches(user_input, data['title'].tolist())

    
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

    user_input = st.text_input("Enter a movie title:")
    if user_input:
        title_of_movie = difflib.get_close_matches(user_input, data['title'])
        if title_of_movie:
            close = title_of_movie[0]
            index_of_movie = data[data.title == close]['index'].values[0]

            # Get similarity scores
            similarity_of_movie = list(enumerate(similarity[index_of_movie]))

            # Sort and display recommendations
            sorted_similarities = sorted(similarity_of_movie, key=lambda x: x[1], reverse=True)
            st.header("Recommended Movies:".title())
            for i in range(10):
                index = sorted_similarities[i+1][0]
                title = data[data.index == index]['title'].values[0]
                st.write(f"{i+1}. {title}")
        else:
            st.write("Movie not found.".title())
    else:
        st.write("Please enter a movie title.")

if __name__ == "__main__":
    main()

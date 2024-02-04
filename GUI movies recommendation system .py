import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import ast
import tmdbsimple as tmdb



# Load data
movies_ds = pd.read_csv(r"C:\Users\Zyad Mohamed\Downloads\archive\tmdb_5000_movies.csv")
credits_ds = pd.read_csv(r"C:\Users\Zyad Mohamed\Downloads\archive\tmdb_5000_credits.csv")

# Merge datasets
movies_ds = movies_ds.merge(credits_ds, on='title')

# Select relevant columns
movies_ds = movies_ds[['movie_id', 'title', 'overview', 'budget', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
movies_ds.dropna(inplace=True)

# Function to convert genres, keywords, and crew to lists
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert2(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except Exception as e:
        print(f"Error converting '{obj}' with error: {e}")
        return []

movies_ds['genres'] = movies_ds['genres'].apply(convert)
movies_ds['keywords'] = movies_ds['keywords'].apply(convert)
movies_ds['cast'] = movies_ds['cast'].apply(lambda x: convert2(x))
movies_ds['crew'] = movies_ds['crew'].apply(lambda x: convert2(x))  # Update: Use convert2 for 'crew' as well

# Concatenate columns to create 'tags'
for column in ['overview', 'budget', 'genres', 'keywords', 'cast', 'crew']:
    movies_ds[column] = movies_ds[column].apply(lambda x: x if isinstance(x, list) else [x])

movies_ds['tags'] = movies_ds['overview'] + movies_ds['budget'] + movies_ds['genres'] + \
                    movies_ds['keywords'] + movies_ds['cast'] + movies_ds['crew']

# Convert 'tags' to lowercase and apply stemming
new_df = movies_ds[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(map(str, x)))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())  # Update: Use lowercase 'x' instead of 'X'

ps = PorterStemmer()
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join([ps.stem(i) for i in x.split()]))

# Vectorize the 'tags' using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vectors)

# Streamlit UI
def main():
    st.title("Movie Recommendation System")

    # User input for movie selection
    selected_movie = st.text_input("Enter the movie title:", "Avatar")

    if st.button("Get Recommendations"):
        # Function to recommend movies
        def recommend(movie):
            movie_index = new_df[new_df['title'] == movie].index[0]
            distances = similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

            recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
            return recommended_movies

        # Display recommendations
        recommendations = recommend(selected_movie)
        st.write(f"Top 5 Recommendations for '{selected_movie}':")
        for rec_movie in recommendations:
            st.write(f"- {rec_movie}")

if __name__ == "__main__":
    main()

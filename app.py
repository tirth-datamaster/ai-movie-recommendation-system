import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

st.title("🎬 AI Movie Recommendation System")

# -----------------------------
# LOAD DATA
# -----------------------------

movies = pd.read_csv("movies_dataset_with_posters.csv")

movies["tags"] = movies["genre"] + " " + movies["language"]

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)

# -----------------------------
# LANGUAGE FILTER
# -----------------------------

st.sidebar.title("📱 Filters")

languages = ["All"] + sorted(movies["language"].unique().tolist())

selected_language = st.sidebar.selectbox(
    "Select Language",
    languages
)

if selected_language != "All":
    filtered_movies = movies[movies["language"] == selected_language]
else:
    filtered_movies = movies

# -----------------------------
# MOVIE BOX UI
# -----------------------------

def movie_box(title):

    st.markdown(
        f"""
        <div style="
            height:250px;
            background-color:black;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:10px;
            color:white;
            font-size:20px;
            text-align:center;
        ">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# SHOW MOVIES
# -----------------------------

def show_movies(movie_df):

    cols = st.columns(5)

    for i in range(min(5, len(movie_df))):

        movie = movie_df.iloc[i]

        with cols[i]:

            movie_box(movie.title)

            st.markdown(
                f"""
                <center>
                <b>{movie.title}</b><br>
                ⭐ {movie.rating}
                </center>
                """,
                unsafe_allow_html=True
            )

            if st.button("View Details", key=f"{movie.title}_{i}"):

                st.session_state["selected_movie"] = movie

# -----------------------------
# TRENDING MOVIES
# -----------------------------

st.subheader("🔥 Trending Movies")

trending = filtered_movies.sort_values(
    by="rating",
    ascending=False
).head(5)

show_movies(trending)

st.divider()

# -----------------------------
# MOVIE RECOMMENDATION
# -----------------------------

def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(i[0])

    return movies.iloc[recommended_movies]

st.subheader("🎥 Movie Recommendation")

selected_movie = st.selectbox(
    "Choose a movie you like",
    filtered_movies["title"].values
)

if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie)

    show_movies(recommendations)

st.divider()

# -----------------------------
# MOOD BASED RECOMMENDATION
# -----------------------------

st.subheader("😊 Mood Based Recommendation")

mood = st.selectbox(
    "How are you feeling today?",
    ["Happy","Excited","Emotional","Romantic","Thrilled"]
)

if st.button("Suggest Movies"):

    if mood == "Happy":
        mood_movies = filtered_movies[
            filtered_movies["genre"].str.contains("Comedy|Family", case=False)
        ]

    elif mood == "Excited":
        mood_movies = filtered_movies[
            filtered_movies["genre"].str.contains("Action|Adventure", case=False)
        ]

    elif mood == "Emotional":
        mood_movies = filtered_movies[
            filtered_movies["genre"].str.contains("Drama", case=False)
        ]

    elif mood == "Romantic":
        mood_movies = filtered_movies[
            filtered_movies["genre"].str.contains("Romance", case=False)
        ]

    elif mood == "Thrilled":
        mood_movies = filtered_movies[
            filtered_movies["genre"].str.contains("Thriller|Crime", case=False)
        ]

    mood_movies = mood_movies.sample(min(5, len(mood_movies)))

    show_movies(mood_movies)

st.divider()

# -----------------------------
# AI CHATBOT
# -----------------------------

st.subheader("🤖 AI Movie Chatbot")

last_movie = st.text_input("Last movie you watched")

chat_mood = st.selectbox(
    "Current Mood",
    ["Happy","Excited","Emotional","Romantic","Thrilled"]
)

if st.button("Ask AI for Suggestions"):

    if last_movie in movies["title"].values:

        recommendations = recommend(last_movie)

        st.write("### AI Suggestions")

        show_movies(recommendations)

    else:

        st.write("Movie not found. Showing mood-based suggestions.")

        mood_movies = filtered_movies.sample(5)

        show_movies(mood_movies)

st.divider()

# -----------------------------
# MOVIE DETAILS
# -----------------------------

if "selected_movie" in st.session_state:

    movie = st.session_state["selected_movie"]

    st.subheader("🎬 Movie Details")

    col1, col2 = st.columns([1,2])

    with col1:

        movie_box(movie.title)

    with col2:

        st.markdown(f"### {movie.title}")

        st.write("⭐ Rating:", movie.rating)

        st.write("🎭 Genre:", movie.genre)

        st.write("🌍 Language:", movie.language)

        st.info("Recommended based on similar genre and language.")
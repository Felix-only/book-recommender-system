import pickle
import streamlit as st
import numpy as np
import pandas as pd

# read csv in dataframe
books = pd.read_csv('data/Books.csv')
ratings = pd.read_csv('data/Ratings.csv')
users = pd.read_csv('data/Users.csv')

nav = st.sidebar.radio("Navigation", ['Home', 'Book Recommender'])

if nav == 'Home':
    st.image('./images/book_recommender.png', width = 300)
    
    st.text(' ')
    st.text(' ')
    st.header('Project Overview')
    st.text(' ')
    st.markdown('''**Advanced Data Analytics:** 
                This project harnesses big data analytics to extract meaningful insights from vast datasets. By analyzing purchasing patterns, 
                browsing behaviors, and user interactions, it uncovers trends and preferences specific to the retailer's customer base.
''')
    st.markdown('''**Customized Recommendation Systems:** 
                At the heart of the project is the development of a state-of-the-art recommendation system. Utilizing cosine similarity, 
                the system will recommend books based on users' past behavior, preferences and reviews. 
                This personalized approach ensures that customers find exactly what they are looking for, and even discover new books that align with their interests.
''')
    st.markdown('''**Visualization:** 
                The project introduces an innovative approach to visualize customer profiles.
                The purpose of visualization is to help companies better to understand how to better market their products.
''')
    st.text(' ')
    st.header('Project Work Flow')
    st.image('images/BigDataFlow.png')

    st.text(' ')
    st.header('Show samples of Data')
    option = st.selectbox(
        'Select a Data source',
        ('books', 'ratings', 'users')
    )

    if option == 'books':
        st.write(books.sample(5))
    if option == 'ratings':
        st.write(ratings.sample(5))
    if option == 'users':
        st.write(users.sample(5))

    # total count images
    st.text(' ')
    st.header('')
   

if nav == 'Book Recommender':
    st.header("Book Recommender")
    model = pickle.load(open('py_objects/model.pkl','rb'))
    book_names = pickle.load(open('py_objects/books_name.pkl','rb'))
    final_rating = pickle.load(open('py_objects/final_rating.pkl','rb'))
    book_pivot = pickle.load(open('py_objects/pivot.pkl','rb'))

    selected_books = st.selectbox(
        "Type or select a book",
        book_names
    )

    def fetch_poster(suggestion):
        book_name = []
        ids_index = []
        poster_url = []

        for book_id in suggestion:
            book_name.append(book_pivot.index[book_id])

        for name in book_name[0]: 
            ids = np.where(final_rating['title'] == name)[0][0]
            ids_index.append(ids)

        for idx in ids_index:
            url = final_rating.iloc[idx]['Image_URL']
            poster_url.append(url)

        return poster_url

    def recommend_book(book_name):
        book_list = []
        book_id = np.where(book_pivot.index == book_name)[0][0]
        # you could appoint n-1 recommendation here
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id].values.reshape(1,-1), n_neighbors=6)
        
        poster_url = fetch_poster(suggestion)

        for i in range(len(suggestion[0])):
            book_list.append(book_pivot.index[suggestion[0][i]])

        return book_list, poster_url

    if st.button('show recommendation'):
        recommendation_books, poster_url = recommend_book(selected_books)
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.text(recommendation_books[1])
            st.image(poster_url[1])

        with col2:
            st.text(recommendation_books[2])
            st.image(poster_url[2])
        
        with col3:
            st.text(recommendation_books[3])
            st.image(poster_url[3])

        with col4:
            st.text(recommendation_books[4])
            st.image(poster_url[4])

        with col5:
            st.text(recommendation_books[5])
            st.image(poster_url[5])

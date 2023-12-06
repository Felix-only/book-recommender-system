import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read csv in dataframe
books = pd.read_csv('data/Books.csv')
ratings = pd.read_csv('data/Ratings.csv')
users = pd.read_csv('data/Users.csv')
final_table = pd.read_csv('data/final_merge_table.csv')

nav = st.sidebar.radio("Navigation", ['Home', 'Book Recommender', 'Visualization'])

# homepage
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
    st.subheader('Data Overview')
    dic = {
        'Books Counts': 271360,
        'Users Counts': 278858,
        'Rating Counts': 1149780
    }
    st.write(dic)
    st.image('images/total_counts.png')

    # GitHub Page
    st.header(' ')
    st.text('-------------------------------------------------------------------------------------------------------------------------------')
    st.markdown("For Details of our project's source codes, check out our [GitHub Page!](https://github.com/Felix-only/book-recommender-system)")
   
#Recommendation page
if nav == 'Book Recommender':
    st.header("Book Recommender")
    with open('py_objects/model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('py_objects/books_name.pkl', 'rb') as file:
        book_names = pickle.load(file)

    final_rating = pd.read_csv('py_objects/final_rating.csv', index_col=0)

    book_pivot = pd.read_csv('py_objects/pivot.csv', index_col=0)

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
    
#visualization page
if nav == 'Visualization':

    tab1, tab2, tab3 = st.tabs(["Ratings & Age Distributions", "Top Rated Books (with Age)", "Top Rated Books (on regions)"])

    with tab1:

        st.header('Data Visualization')
        st.text(' ')
        st.markdown('**Chcek out the distribution of Data**')
        st.text(' ')
        with st.container():
            # Create a figure for Matplotlib
            fig, ax = plt.subplots()

            # Create the histogram with a log scale for the y-axis
            sns.histplot(ratings['Book-Rating'], kde=False, bins=30, ax=ax)
            ax.set_yscale('log')  # Sets a logarithmic scale on the y-axis
            ax.set(xlabel='Book Rating', ylabel='Log Count')  # Labels the axes

            # Adds a KDE plot on top of the histogram with the same x-axis
            sns.kdeplot(ratings['Book-Rating'], ax=ax, color='red')

            # Sets the title for the plot
            ax.set_title('Distribution of Book Ratings')

            # Display the plot in the Streamlit app
            st.pyplot(fig)

            st.header(' ')

        with st.container():
            # Create a figure for Matplotlib
            fig, ax = plt.subplots()

            # Create the histogram and KDE plot
            sns.histplot(users['Age'], kde=True)
            ax.set_xlim(5, 80)  # Sets the x-axis limit from 5 to 80
            ax.set_xlabel('Age')  # Sets the x-axis label
            ax.set_ylabel('Count')  # Sets the y-axis label
            ax.set_title('Age Distribution of Users')  # Sets the title of the plot

            # Display the plot in the Streamlit app
            st.pyplot(fig)

    with tab2:

        # ratings_books_50plus_df = pd.read_csv('../data/final_rating.csv', index_col=0).reset_index().drop(columns=['index'])
        ratings_books_50plus_df = final_table.copy()

        st.header('Top Rated books')
        # # Count the number of ratings for each book
        # book_rating_counts = ratings['ISBN'].value_counts().reset_index()
        # book_rating_counts.columns = ['ISBN', 'Rating-Count']

        # # Filter for books with 50 or more ratings
        # books_with_50plus_ratings = book_rating_counts[book_rating_counts['Rating-Count'] >= 50]

        # # Merge this filtered dataset with the original ratings and then with the books dataset
        # ratings_50plus_df = pd.merge(books_with_50plus_ratings, ratings, on="ISBN")
        # ratings_books_50plus_df = pd.merge(ratings_50plus_df, books, on="ISBN")

        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        num = st.slider('Top Rated Books (with at least 30 ratings)', 0, 20, 5)
        # Top num rated books with 30+ ratings
        top_rated_books_50plus_df = ratings_books_50plus_df.groupby('title')['rating'].mean().reset_index()
        top_rated_books_50plus_df = top_rated_books_50plus_df.sort_values('rating', ascending=False).head(num)

        # Visualization for Top num Rated Books and Their Ratings
        fig = plt.figure(figsize=(10, num + 1))
        chart = sns.barplot(x='rating', y='title', data=top_rated_books_50plus_df, palette='coolwarm')
        plt.title('Top '+ str(num) + ' Rated Books (with at least 30 ratings)')
        plt.xlabel('Average Rating')
        plt.ylabel('Book Title')

        # Adding the text on the bars
        for p in chart.patches:
            width = p.get_width()
            plt.text(p.get_width(), p.get_y() + p.get_height()/2. + 0.1, '{:1.2f}'.format(width), ha="left")
        
        st.pyplot(fig)


        age_young = st.slider('Top Rated Books with Younger Users (with at least 30 ratings)', 0, 20, 5)

        # Calculate the average age of users for each book with 30+ ratings
        # ratings_books_users_50plus_df = pd.merge(ratings_books_50plus_df, users, on="User-ID")
        mean_age = ratings_books_50plus_df['Age'].mean()
        young_df = ratings_books_50plus_df[ratings_books_50plus_df['Age'] < mean_age]

        books_avg_age_50plus_df = young_df.groupby('title')['rating'].mean().reset_index()

        # Top books with the youngest average user age
        top_5_youngest_books_50plus_df = books_avg_age_50plus_df.sort_values('rating',ascending=False).head(age_young)

        # Visualization for Top Books with the Youngest Average User Age
        fig2 = plt.figure(figsize=(10, age_young + 1))
        chart = sns.barplot(x='rating', y='title', data=top_5_youngest_books_50plus_df, palette='mako')
        plt.title('Top ' + str(age_young) + ' Books with the Youngest Average User Age (with at least 30 ratings)')
    
        plt.ylabel('Book Title')

        # Adding the text on the bars
        for p in chart.patches:
            width = p.get_width()
            plt.text(p.get_width(), p.get_y() + p.get_height()/2. + 0.1, '{:1.2f}'.format(width), ha="left")

        st.pyplot(fig2)


        age_old = st.slider('Top Rated Books with Older Users (with at least 30 ratings)', 0, 20, 5)
        
        # Top books with the oldest average user age
        old_df = ratings_books_50plus_df[ratings_books_50plus_df['Age'] >= mean_age]

        books_avg_age_old_df = old_df.groupby('title')['rating'].mean().reset_index()
        top_5_oldest_books_50plus_df = books_avg_age_old_df.sort_values('rating', ascending=False).head(age_old)

        # Visualization for Top Books with the Oldest Average User Age
        fig3 = plt.figure(figsize=(10, age_old + 1))
        chart = sns.barplot(x='rating', y='title', data=top_5_oldest_books_50plus_df, palette='copper')
        plt.title('Top ' + str(age_old) + ' Books with the Oldest Average User Age (with at least 30 ratings)')
        
        plt.ylabel('Book Title')

        # Adding the text on the bars
        for p in chart.patches:
            width = p.get_width()
            plt.text(p.get_width(), p.get_y() + p.get_height()/2. + 0.1, '{:1.2f}'.format(width), ha="left")

        st.pyplot(fig3)

    with tab3:

        st.header('Top rated books in regions')

        with open('py_objects/region_unique.pkl', 'rb') as file:
            region_unique = pickle.load(file)
    
        selected_region = st.selectbox(
            "Please select the region",
            region_unique
        )
        
        num_top = st.slider('please select the number of books', 0, 10, 5)

        ratings_books_us = final_table[final_table['country'] == selected_region]

        # # Filter users from the US and the UK
        # users_us = users[users['Location'].str.contains(selected_region, case=False, na=False)]

        # # Merge the filtered users with the ratings
        # ratings_us = pd.merge(users_us, ratings, on="User-ID")

        # # Merge this with the books data
        # ratings_books_us = pd.merge(ratings_us, books, on="ISBN")

        # Count the number of ratings for each book in the US and the UK
        book_rating_counts_us = ratings_books_us['ISBN'].value_counts().reset_index()
        book_rating_counts_us.columns = ['ISBN', 'Rating-Count']

        # Merge this filtered dataset with the original ratings and then with the books dataset
        ratings_books_50plus_us = pd.merge(book_rating_counts_us, ratings_books_us, on="ISBN")

        # Top rated books in the US and the UK with 50+ ratings
        top_rated_books_50plus_us = ratings_books_50plus_us.groupby('title')['rating'].mean().reset_index()
        top_rated_books_50plus_us = top_rated_books_50plus_us.sort_values('rating', ascending=False).head(num_top)

        # Visualization for Top Rated Books in the US and the UK with 50+ ratings
        fig4 = plt.figure(figsize=(10, num_top + 1))
        chart = sns.barplot(x='rating', y='title', data=top_rated_books_50plus_us, palette='coolwarm')
        plt.title('Top ' + str(num_top) + ' Rated Books in the ' + str(selected_region))
        plt.xlabel('Average Rating')
        plt.ylabel('Book Title')

        # Adding the text on the bars
        for p in chart.patches:
            width = p.get_width()
            plt.text(p.get_width(), p.get_y() + p.get_height()/2. + 0.2, '{:1.2f}'.format(width), ha="left")
        
        st.pyplot(fig4)
    
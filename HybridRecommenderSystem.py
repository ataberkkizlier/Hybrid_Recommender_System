'''Make 10 movie recommendations for the user whose ID is given, using the item-based and user-basedrecommender
methods'''

'''Data Set Story: The dataset is provided byMovieLenstar, a movie recommendation service.It contains ratings scores 
for movies along with the movies themselves.It contains 2,000,0263 ratings on 27,278 movies.This dataset was created 
on October 17, 2016.138 .493 users and data from January 09, 1995 to March 31, 2015. Users were selected randomly, 
and it was found that all selected users voted for at least 20 movies.'''


'''Task 1: :  Data Preparation

Step 1: Read the movie, rating datasets'''

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movie = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/movie.csv")
rating = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/rating.csv")
movie.head()
rating.head()
'''Step 2: Add the movie names of the IDs to the rating dataset and the 
product from the movie dataset. '''
df= movie.merge(rating, how="left", on='movieId')
df.head()
'''Step3: List the names of the movies with a total number of ratings below 1000 and 
remove them from the dataset.'''
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()
rare_movies = comment_counts[comment_counts["count"] <= 1000].index

common_movies = df[~df['title'].isin(rare_movies)]
common_movies.head()
common_movies["title"].nunique() #3159

'''Step4: Create a pivot table for the dataframe with the userIDs in the index, movie names 
in the columns, and ratings in the columns.'''
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
#(138493, 3159)

'''Step5: Functionalize all the operations.'''

user_movie_df.head(10)

#example:
movie_name = "Matrix, The (1999)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/movie.csv')
    rating = pd.read_csv('/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


'''Task 2: Determining the Movies Watched by the User for Recommendation'''

'''Step 1: Select a random userid'''
#generate a random user ID from the user_movie_df DataFrame.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user = 28941

'''Step 2: Create a new frame with the name random_user_df consisting of the observation units of the selected user'''
#Check what movies did the user 31838 watch
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df

'''Step3: Add the selected users to a list called movies_watched'''

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched



# NUmber of movies watched by our random user.
len(movies_watched) #33
'''Task 3: Accessing Data and IDs of Other Users Watching the Same Movies'''
'''Step 1: Select the boxes for the movies watched by the selected user from the user_movie_df and create a 
newdataframe called movies_watched_df.'''

#Double check.
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Sabrina (1995)"]
'''title    Sabrina (1995)
userId                 
28941.0             5.0'''

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape
#(138493, 33)

'''Step 2: Create a new dataframe called user_movie_count with the information of how 
many movies each user has watched by the selected user.'''
user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]
user_movie_count.head()
'''   userId  movie_count
0     1.0            1
1     2.0            2
2     3.0            4
3     4.0            6
4     5.0           11'''

'''Step3: Create a list called users_same_movies with the username 
of the users who watched 60% or more of the movies the selected user voted for'''
#1 user watched 978 movies.
user_movie_count[user_movie_count["movie_count"] ==33].count()
'''userId         17
movie_count    17'''

percentage = len(movies_watched) * 60 / 100
percentage
#19.8


users_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]
users_same_movies.count()
users_same_movies.shape
#4139
'''Task 4: Identification of the Most Similar Users with the User to Make a Recommendation'''
'''Step 1: Filter the movies_watched_dfdataframe to include the ids of users that are similar to the selected user in the user_same_movies list.'''
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

final_df.shape #(4140,33)
final_df.T.corr()
'''Step 2: Create a new corr_df dataframe with the correlations of the users with each other.'''
final_df = final_df.drop_duplicates()
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()
'''   user_id_1  user_id_2      corr
0   105664.0    78262.0 -0.954490
1    78262.0   105664.0 -0.954490
2   119715.0    55005.0 -0.952579
3    55005.0   119715.0 -0.952579
4   137558.0   104652.0 -0.950000'''

'''Step3: Create a newdataframe called top_users by filtering the users with high correlation (above 0.65) with the selected user.'''

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()
'''      userId      corr
64   28941.0  1.000000
63   13477.0  0.802181
62   45158.0  0.800749
61  101628.0  0.790405
60    7542.0  0.772183'''


'''Step4: Create the top_users dataframe with the retrieving dataset'''
rating = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/rating.csv")
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head()

'''     userId      corr  movieId  rating
33  13477.0  0.802181        1     4.0
34  13477.0  0.802181        2     3.0
35  13477.0  0.802181        3     2.0
36  13477.0  0.802181        5     3.0
37  13477.0  0.802181        7     5.0'''


'''Task 5: Calculating the Weighted Average Recommendation Score and Retaining the Top 5 Films'''

'''Step 1: Create a new variable called weighted_rating which is the product of each user's corrve rating.'''

'''Step 2: 
Create a new dataframe called recommendation_df which contains the movie id and the average value of all users' 
weighted ratings for each movie.'''

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
'''         weighted_rating
movieId                 
1               2.506655
2               1.684571
3               1.638914
4               1.691054
5               1.483689
                  ...
112183          2.028250
112370          1.314665
112552          2.366291
112946          2.957996
118696          2.300664
[5990 rows x 1 columns]'''

'''Step3: In the recommendation_df, select the movies with a weighted rating greater 
than 3.5 and sort them according to the weighted rating.'''
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()
'''   movieId  weighted_rating
0        1         2.506655
1        2         1.684571
2        3         1.638914
3        4         1.691054
4        5         1.483689'''

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

'''Step4: From the movie dataset, enter the movie names and 
select the first 5 movies to recommend.'''

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

'''      movieId  weighted_rating
47         53         3.952023
667       887         3.680055
1367     1922         3.763580
1485     2057         3.763580
1819     2485         3.763580
1943     2675         3.680055
2259     3118         3.763580
3952     6216         3.664714
4557     7585         3.664714
4571     7669         3.522390'''


movies_to_be_recommend.merge(movie[["movieId", "title"]])

'''   movieId  weighted_rating                                              title
0       53         3.952023                                    Lamerica (1994)
1     1922         3.763580                                    Whatever (1998)
2     2057         3.763580                     Incredible Journey, The (1963)
3     2485         3.763580                              She's All That (1999)
4     3118         3.763580                                 Tumbleweeds (1999)
5      887         3.680055                              Talk of Angels (1998)
6     2675         3.680055  Twice Upon a Yesterday (a.k.a. Man with Rain i...
7     6216         3.664714     Nowhere in Africa (Nirgendwo in Afrika) (2001)
8     7585         3.664714                                  Summertime (1955)
9     7669         3.522390                         Pride and Prejudice (1995)'''

#Let's see the top 5 movies:
movies_to_be_recommend.merge(movie[["movieId", "title"]])[:5]
'''   movieId  weighted_rating                           title
0       53         3.952023                 Lamerica (1994)
1     1922         3.763580                 Whatever (1998)
2     2057         3.763580  Incredible Journey, The (1963)
3     2485         3.763580           She's All That (1999)
4     3118         3.763580              Tumbleweeds (1999)'''


# Make an item-based suggestion based on the name of the movie that the user has watched with the highest score.

# ▪ 5 suggestions user-based
# ▪ 5 suggestions item-based

movie = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/movie.csv")
rating = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 5/HybridRecommender-221114-235254/datasets/rating.csv")
# The last highly-rated movie by user 108170:

user = 108170
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]
movie_id
#7044

# ▪ 5 suggestions user-based
movies_to_be_recommend.merge(movie[["movieId", "title"]])[:5]['title'].to_list()
'''['Lamerica (1994)',
 'Whatever (1998)',
 'Incredible Journey, The (1963)',
 "She's All That (1999)",
 'Tumbleweeds (1999)']'''

# ▪ 5 suggestions item-based
movie_name = movie[movie['movieId'] == movie_id]['title'].values[0]
movie_name = user_movie_df[movie_name]
moveis_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False)
moveis_from_item_based[1:6].index.to_list()
'''['My Science Project (1985)',
 'Mediterraneo (1991)',
 'Old Man and the Sea, The (1958)',
 "National Lampoon's Senior Trip (1995)",
 'Clockwatchers (1997)']'''
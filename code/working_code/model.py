#import libraries
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from pathlib import Path
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np
from itertools import islice

#set up paths
PATH = Path(r'././')
DATA= PATH/'data'
CODE= PATH/'code'
PRODUCTS=PATH/'products'
WORKING_DATA= DATA/'working_data'

reader = Reader(rating_scale=(1, 5))
df = pd.read_csv(WORKING_DATA/'ratings_small.csv')
links = pd.read_csv(WORKING_DATA/'links_with_title.csv')
links_ratings =  pd.read_csv(WORKING_DATA/'ratings_with_links.csv')
data = Dataset.load_from_df(df[['userId','movieId','rating']], reader=reader)



def take(n, iterable):
    """Return first n items of the iterable as a list taken from itertools recipes"""
    return list(islice(iterable, n))

def add_user(ratings,df):
    """
    Accepts,dataframe, movies and ratings in a dictionary format{movie:rating}
    returns dataset
    """
    user = df['userId'].max()+1
    for movie in ratings:
        df = df.append(pd.Series([user,movie,ratings[movie]]),index=df.columns,ignore_index=True)
        data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader=reader)
    return data

def train_model(data):
    """
    Accepts dataset and returns trained model
     """
    trainsetfull = data.build_full_trainset()
    algo = SVD()
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    algo.fit(trainsetfull)
    return algo


def make_predictions(user,df,model,n):
    """ Accepts user and dictionary containing movies
    returns n movies that the user might like above a threshold"""
    movie_predictions = {}
    for movie in df['movieId']:
        #if user has not seen movie
        if len(df[(df['userId'] == user)  & (df['movieId'] == movie)]) == 0:
            prediction = model.predict(uid = user, iid = movie)
            movie_predictions[movie]=prediction.est
    movie_predictions = {k: v for k, v in sorted(movie_predictions.items(), key=lambda item: item[1],reverse=True)}
    suggested = take(n, movie_predictions.items())
    return suggested
def id_to_title(movieIds,df):
    """accepts movie id and returns their title uses links_with_title csv """
    titles = []
    for movieId in movieIds:
       titles.append(df[df['movieId']==movieId].iloc[0]['original_title'])
    return titles
def title_to_id(titles,df):
    """accepts movie title and returns id uses links_with_title csv"""
    movieIds = []
    for title in titles:
       movieIds.append(df[df['original_title']==title].iloc[0]['movieId'])
    return movieIds
## Suggested use
# data=add_user()
#model=train_model(data)
##get n movie suggestions
#suggestions = np.array(make_predictions(user,df,model,n_movies))
##select the movie id
#suggested_movie_id = suggestions[:,0]
##return titles to user
#titles = id_to_title(suggested_movie_id,links)

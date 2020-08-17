from flask import Flask
import model as m
import pandas as pd 
import numpy as np 
from pathlib import Path 
import time

#set up paths
PATH = Path(r'../../')
DATA= PATH/'data'
CODE= PATH/'code'
PRODUCTS=PATH/'products'
WORKING_DATA= DATA/'working_data'

df = pd.read_csv(WORKING_DATA/'ratings_small.csv')
links = pd.read_csv(WORKING_DATA/'links_with_title.csv')
links_ratings =  pd.read_csv(WORKING_DATA/'ratings_with_links.csv')
app = Flask(__name__)


def get_seconds(start,end):
    duration = round((end-start)/60)
    seconds =60*(((end-start)/60) - duration )
    return seconds

@app.route('/')
def home():
    start = time.time()
    ratings = {}
    movieId = m.title_to_id(['Toy Story','Toy Story 2',"Toy Story 3","The Lion King","A Bug's Life","The Hunchback of Notre Dame"],links)
    for movie in movieId:
        ratings[movie]=5
    data=m.add_user(ratings,df)
    model=m.train_model(data)
    #get n movie suggestions
    user = df['userId'].max()
    n_movies=5
    suggestions = np.array(m.make_predictions(user,df,model,n_movies))
    #select the movie id
    suggested_movie_id = suggestions[:,0]
    #return titles to user
    titles = m.id_to_title(suggested_movie_id,links)
    suggested_titles={}
    suggested_titles['Suggested Titles']= titles
    suggested_titles['Watched Titles']= ['Toy Story','Toy Story 2',"Toy Story 3","The Lion King","A Bug's Life","The Hunchback of Notre Dame"]
    end = time.time()
    minutes = round((end-start)/60)
    seconds = get_seconds(start,end)
    suggested_titles['minutes']=minutes
    suggested_titles['seconds']=seconds
    return suggested_titles




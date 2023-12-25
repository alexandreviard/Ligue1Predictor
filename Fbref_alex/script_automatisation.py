from scrapping import *
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from fonctions_preprocess import *

def automatisation():

    ajd = datetime.now()
    df = pd.read_csv("Projet-python/Fbref_alex/SOCCER_201223_18h-2 copy.csv", index_col=0)
    df2 = preprocess(df.copy())
    latest_date = df2['DateTime'].max()
    difference = (ajd - latest_date).days

    A = scrape_latest_ligue1_data()
    df = preprocess(add_new_matches(df, A[0]))

    latest_date2 = df['DateTime'].max()

    ZZ = A[1]
    ZZ == ZZ[ZZ['DateTime'] >= latest_date2].dropna(subset=["DateTime"])
    


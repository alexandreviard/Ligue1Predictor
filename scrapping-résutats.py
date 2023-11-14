#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests

#on cherche à scrapper les données de classement à chaque journée de la saison 2022/2023
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2022-2023"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

#on isole le tableau qui nous intéresse
classement = soup.find("table", class_="wikitable")
#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests

#on cherche à scrapper les données de classement à chaque journée de la saison 2022/2023
url = "https://www.footballcritic.com/ligue-1/season-2022-2023/matches/4/65453"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

#on isole le tableau qui nous intéresse
résultats = soup.find("span", class_="rounds r_67146 w_38")
colonnes = ["Journée", "Domicile", "Extérieur", "Buts domicile", "Buts extérieur", "Résultat"]
résultats_22_23 = pd.DataFrame(columns=colonnes)


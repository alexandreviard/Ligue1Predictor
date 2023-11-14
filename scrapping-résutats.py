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

equipes = résultats.select('.info-box .text.hidden-xs a')
equipes_final =[]
for element in equipes:
    equipes_final.append(element.text.strip())

equipes_domicile = equipes_final[::2]
equipes_exterieure = equipes_final[1::2]

scores = résultats.find_all('span', class_='score-text')
scores_int =[]
for element in scores:
    element2 = element.text.strip()
    scores_int.append(element2.split(' - '))
    scores_final = [item for sublist in scores_int for item in sublist]

scores_domicile = scores_final[::2]
scores_exterieure = scores_final[1::2]


résultats_22_23 = pd.DataFrame()
résultats_22_23['Journée'] = [1]*10
résultats_22_23['Domicile'] = equipes_domicile
résultats_22_23['Extérieur'] = equipes_exterieure
résultats_22_23['Buts domicile'] = scores_domicile
résultats_22_23['Buts extérieur'] = scores_exterieure
résultats_22_23['Résultat'] = 'E'
résultats_22_23.loc[résultats_22_23['Buts domicile'] > résultats_22_23['Buts extérieur'], 'Résultat'] = 'D'
résultats_22_23.loc[résultats_22_23['Buts domicile'] == résultats_22_23['Buts extérieur'], 'Résultat'] = 'N'
print(résultats_22_23)
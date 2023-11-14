#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests


#on cherche à scrapper les données de classement à chaque journée de la saison 2022/2023
url = "https://www.footballcritic.com/ligue-1/season-2022-2023/matches/4/65453"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")
noms_colonnes = ['Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
résultats_22_23_finaux = pd.DataFrame(columns=noms_colonnes)
#on effectue notre requête sur les 38 journées de Ligue 1
#on remarque qu'il y a un soucis à la journée 2. On divise donc en trois parties 
#la J1
for i in range(1,2):
    #on isole le tableau qui nous intéresse
    résultats = soup.find("span", class_="rounds r_67146 w_" + str(i))

    #on construit la liste des équipes jouant à domicile, et celle des équipes jouant à l'extérieur correspondante
    equipes = résultats.select('.info-box .text.hidden-xs a')
    equipes_final =[]
    for element in equipes:
        equipes_final.append(element.text.strip())

    equipes_domicile = equipes_final[::2]
    equipes_exterieure = equipes_final[1::2]

    #on construit la liste des buts marqués à domicile, et celle des buts marqués à l'extérieur correspondante
    scores = résultats.find_all('span', class_='score-text')
    scores_int =[]
    for element in scores:
        element2 = element.text.strip()
        scores_int.append(element2.split(' - '))
        scores_final = [item for sublist in scores_int for item in sublist]

    scores_domicile = scores_final[::2]
    scores_exterieure = scores_final[1::2]

    #on crée un data frame avec les résultats
    noms_colonnes = ['Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
    résultats_22_23 = pd.DataFrame(columns=noms_colonnes)
    résultats_22_23['Journée']= [i]*10
    résultats_22_23['Domicile'] = equipes_domicile
    résultats_22_23['Extérieur'] = equipes_exterieure
    résultats_22_23['Buts domicile'] = scores_domicile
    résultats_22_23['Buts extérieur'] = scores_exterieure
    résultats_22_23['Résultat'] = 'E'
    résultats_22_23.loc[résultats_22_23['Buts domicile'] > résultats_22_23['Buts extérieur'], 'Résultat'] = 'D'
    résultats_22_23.loc[résultats_22_23['Buts domicile'] == résultats_22_23['Buts extérieur'], 'Résultat'] = 'N'
    résultats_22_23_finaux = résultats_22_23_finaux._append(résultats_22_23, ignore_index=True)

#la J2
for i in range(2,3):
    résultats = soup.find("span", class_="rounds r_67146 w_" + str(i))

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

    noms_colonnes = ['Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
    résultats_22_23 = pd.DataFrame(columns=noms_colonnes)
    résultats_22_23['Journée']= [i]*9
    résultats_22_23['Domicile'] = equipes_domicile
    résultats_22_23['Extérieur'] = equipes_exterieure
    résultats_22_23['Buts domicile'] = scores_domicile
    résultats_22_23['Buts extérieur'] = scores_exterieure
    résultats_22_23['Résultat'] = 'E'
    résultats_22_23.loc[résultats_22_23['Buts domicile'] > résultats_22_23['Buts extérieur'], 'Résultat'] = 'D'
    résultats_22_23.loc[résultats_22_23['Buts domicile'] == résultats_22_23['Buts extérieur'], 'Résultat'] = 'N'
    #on rajoute le résultat qu'il nous manquait
    manquant = {'Journée':'2','Domicile' : 'Lorient', 'Extérieur':'Lyon','Buts domicile':'3','Buts extérieur':'1','Résultat':'D'}
    résultats_22_23 = résultats_22_23._append(manquant, ignore_index=True)
    résultats_22_23_finaux = résultats_22_23_finaux._append(résultats_22_23, ignore_index=True)

for i in range(3,39):
    résultats = soup.find("span", class_="rounds r_67146 w_" + str(i))

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

    noms_colonnes = ['Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
    résultats_22_23 = pd.DataFrame(columns=noms_colonnes)
    résultats_22_23['Journée']= [i]*10
    résultats_22_23['Domicile'] = equipes_domicile
    résultats_22_23['Extérieur'] = equipes_exterieure
    résultats_22_23['Buts domicile'] = scores_domicile
    résultats_22_23['Buts extérieur'] = scores_exterieure
    résultats_22_23['Résultat'] = 'E'
    résultats_22_23.loc[résultats_22_23['Buts domicile'] > résultats_22_23['Buts extérieur'], 'Résultat'] = 'D'
    résultats_22_23.loc[résultats_22_23['Buts domicile'] == résultats_22_23['Buts extérieur'], 'Résultat'] = 'N'
    résultats_22_23_finaux = résultats_22_23_finaux._append(résultats_22_23, ignore_index=True)
print(résultats_22_23_finaux)
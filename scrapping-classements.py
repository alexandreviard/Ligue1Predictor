#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

def fonction_classement(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    annee = re.search(r'\d{4}-\d{4}', url).group()
    if annee == "2009-2010":
        supplement = " centre"
    else:
        supplement = " center"
    if annee == "2015-2016" or annee=="2014-2015":
        supplement2 = " nowrap"
    else:
        supplement2 = ""

    
    tableau = soup.find("table", class_="wikitable sortable" + str(supplement) + str(supplement2))
    n = tableau.get_text().count("[r ") + tableau.get_text().count("[a ") + tableau.get_text().count("[e ")
    print(n)
    balises_td = tableau.find_all('td')
    m = len(balises_td)
    equipes =[]
    balises_a = tableau.find_all('a')
    for balise in balises_a[n:]:
        equipes.append(balise.text)
    
    #en raison du Covid, le nombre de journée n'est pas le même en 2019-2020
    if annee == "2019-2020":
        y=29
    else:
        y=39

    #on crée un dataframe#
    colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, y)]
    classement = pd.DataFrame(columns=colonnes)

    #on y ajoute les classements pour chaque journée
    if annee == "2021-2022":
        for i in range(5):
            liste = [x.get_text(strip=True) for x in balises_td[1+42*i:39+42*i]]
            liste = [equipes[i]] + liste
            classement.loc[len(classement)] = liste
        for i in range(5,20):
            liste = [x.get_text(strip=True) for x in balises_td[0+42*i:38+42*i]]
            liste = [equipes[i]] + liste
            classement.loc[len(classement)] = liste
    else: 
        for i in range(20):
            liste = [x.get_text(strip=True) for x in balises_td[1+int(m/20)*i:y+int(m/20)*i]]
            liste = [equipes[i]] + liste
            classement.loc[len(classement)] = liste
    
    if annee == "2019-2020":
        for i in range(29,39):
            nom_colonne = "J" + str(i)
            classement[nom_colonne] = 0
    return classement

liste_url = []

for annee in range(2009, 2023):
    lien = f"https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_{annee}-{annee+1}"
    liste_url.append(lien)

noms_colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
dataframe_classement = pd.DataFrame(columns=noms_colonnes)
#on ajoute les résultats pour chaque saison, en vérifiant qu'il n'y a pas d'erreur
for element in liste_url:
    try:
        classement = fonction_classement(element)
        dataframe_classement = dataframe_classement._append(classement, ignore_index=True)
    except Exception as e:
        print(f"Erreur à l'élément {element}: {e}")


    
#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

#on crée une première fonction pour extraire les données des tableaux Wikipedia montrant l'évolution du classement de chaque équipe à chaque saison 
def fonction_classement(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    #on extrait le paragraphe du code HTML où se trouve nos informations
    #pour certaines année le marqueur du paragraphe n'est pas le même. On crée donc des conditions.
    annee = re.search(r'\d{4}-\d{4}', url).group()
    if annee == "2009-2010":
        supplement = " centre"
    else:
        supplement = " center"
    if annee == "2015-2016" or annee=="2014-2015":
        supplement2 = " nowrap"
    else:
        supplement2 = ""

    #on extrait le paragraphe
    tableau = soup.find("table", class_="wikitable sortable" + str(supplement) + str(supplement2))

    #on extrait le nom des équipes
    n = tableau.get_text().count("[r ") + tableau.get_text().count("[a ") + tableau.get_text().count("[e ")
    equipes =[]
    balises_a = tableau.find_all('a')
    for balise in balises_a[n:]:
        equipes.append(balise.text)

    #on va maintenant extraire les classements pour chaque journée
    balises_td = tableau.find_all('td')
    m = len(balises_td)
    
    #en raison du Covid, le nombre de journée n'est pas le même en 2019-2020
    if annee == "2019-2020":
        y=29
    else:
        y=39

    #on crée un dataframe#
    colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, y)]
    classement = pd.DataFrame(columns=colonnes)

    #on y ajoute les classements pour chaque journée
    #on a une petite anomalie en 2021-2022, avec un décalage des balises à partir d'une certaine équipe. On crée donc une condition :
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
    
    #pour l'année Covid, on met 0 comme classement pour les journées qui n'ont pas eu lieu
    if annee == "2019-2020":
        for i in range(29,39):
            nom_colonne = "J" + str(i)
            classement[nom_colonne] = 0
    
    #on ajoute une colonne avec la Saison
    classement.insert(0,"Saison", str(annee))
    return classement

#Le tableau de wikipedia était très pratique pour scraper les données. Cependant nous nous sommes rendus compte en cours de route qu'il n'y en avait pas pour les saisons antérieur à 2009-2010
#on crée donc une deuxième fonction pour scraper les données sur un nouveau site
#sur ce site on a le classement par journée, par saison 
#il y a donc beaucoup plus de pages à scraper.
#nous avons décidé de garder les deux fonctions plutot que de tout scraper sur ce site
def fonction_classement2(url):
    df_classement = pd.DataFrame()
    annee = re.search(r'\d{4}-\d{4}', url).group()
    #on crée l'url pour chaque journée et on extrait le code HTML
    for i in range(1,39):
        url2 = url + str(i)
        page = requests.get(url2)
        soup = BeautifulSoup(page.text, "html.parser")

        #on extrait le paragraphe HTML qui nous intéresse
        tableau = soup.find("table", class_="tableau")

        #on extrait le paragraphe avec les noms des équipes et on crée une liste avec ces derniers
        equipes = tableau.select('tr.classement td.gras, tr.classement-avec-separateur td.gras')
        equipes_final = []
        for equipe in equipes:
            equipes_final.append(equipe.get_text(strip=True))
        
        #on extrait le paragraphe avec les classements des équipes et on crée une liste avec ces derniers
        classements = tableau.select('tr.classement span.flag-place, tr.classement-avec-separateur span.flag-place')
        classement_final = []
        for classement in classements:
            #sur ce site si deux équipes ont le même classement, un tiret est affiché. on crée donc une condition
            if classement.get_text(strip=True) == "-":
                classement_final.append(classement_final[-1])
            else:
                classement_final.append(classement.get_text(strip=True))
        dictionnaire_classement = dict(zip(equipes_final, classement_final))
        #on crée la colonne avec le nom des équipes pour la première journée, ensuite on ajoute seulement le classement
        if i==1:
            df_classement["Équipes"] = equipes_final
            df_classement["J" + str(i)] = df_classement['Équipes'].map(dictionnaire_classement)
        else:
            df_classement["J" + str(i)] = df_classement['Équipes'].map(dictionnaire_classement)
    
    #on rajoute la colonne avec la saison
    df_classement.insert(0,"Saison", str(annee))
    return(df_classement)

#on crée les url des deux sites

liste_url = []

for annee in range(2009, 2023):
    lien = f"https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_{annee}-{annee+1}"
    liste_url.append(lien)

liste_url2 = []

for annee in range(2002, 2009):
    lien = f"https://www.deux-zero.com/ligue-1/classement-general/edition/{annee}-{annee+1}/init/1/fin/"
    liste_url2.append(lien)

#on crée le dataframe final
noms_colonnes = ["Saison", "Équipes"] + ["J{}".format(i) for i in range(1, 39)]
dataframe_classement = pd.DataFrame(columns=noms_colonnes)
#on ajoute les résultats pour chaque saison, en vérifiant qu'il n'y a pas d'erreur
for element in liste_url2:
    try:
        classement = fonction_classement2(element)
        dataframe_classement = dataframe_classement._append(classement, ignore_index=True)
    except Exception as e:
        print(f"Erreur à l'élément {element}: {e}")

for element in liste_url:
    try:
        classement = fonction_classement(element)
        dataframe_classement = dataframe_classement._append(classement, ignore_index=True)
    except Exception as e:
        print(f"Erreur à l'élément {element}: {e}")

#on enregistre le dataframe final au format csv
dataframe_classement.to_csv('dataframe_classements.csv', encoding = 'utf-8', index=False)
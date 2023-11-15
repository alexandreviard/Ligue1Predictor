#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

#on créé une fonction qui permettent de scrapper les résultats de chaque journée, en prenant en entrée un url et un nom de classe HTML
def fonction_résultats(url,classe):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    noms_colonnes = ['Saison','Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
    résultats_finaux = pd.DataFrame(columns=noms_colonnes)
    url_parts = url.split("/")
    annee_part = [part for part in url_parts if "season" in part][0]
    annee = annee_part.replace("season-", "")
    if annee == "2019-2020":
        x=29
    else:
        x=39
    
    for i in range(1,x):
        tableau = soup.find_all("span", class_="rounds r_" + str(classe) + " w_" + str(i))
        equipes = []
        for element in tableau:
            equipes.append(element.select('.info-box .text.hidden-xs a'))
        equipes_finales =[]
        for equipe in equipes:
            for balise_a in equipe:
                texte_equipe = BeautifulSoup(str(balise_a), 'html.parser').get_text()
                equipes_finales.append(texte_equipe)
        equipes_finales = [equipe.strip() for equipe in equipes_finales]

        equipes_domicile = equipes_finales[::2]
        equipes_exterieure = equipes_finales[1::2]

        scores = []
        for element in tableau:
            scores.append(element.select('.btn-info'))
        scores_int =[]
        for element in scores:
            scores_matche = re.findall(r'>-<|\d+\s*-\s*\d+|>20:45<|>21:00<|>17:00<',str(element))
            scores_int.extend(scores_matche) 
        scores_int = ['0 - 0' if item in ('>-<','>20:45<','>21:00<','>17:00<') else item for item in scores_int]  
        scores_int = [element.split(' - ') for element in scores_int]
        scores_finaux = [item for sublist in scores_int for item in sublist]
        
        scores_domicile = scores_finaux[::2]
        scores_exterieure = scores_finaux[1::2]

        résultats = pd.DataFrame(columns=noms_colonnes)
        url_parts = url.split("/")
        annee_part = [part for part in url_parts if "season" in part][0]
        annee = annee_part.replace("season-", "")
        résultats['Saison']=[annee]*10
        résultats['Journée']= [i]*10
        résultats['Domicile'] = equipes_domicile
        résultats['Extérieur'] = equipes_exterieure
        résultats['Buts domicile'] = scores_domicile
        résultats['Buts extérieur'] = scores_exterieure
        résultats['Résultat'] = 'E'
        résultats.loc[résultats['Buts domicile'] > résultats['Buts extérieur'], 'Résultat'] = 'D'
        résultats.loc[résultats['Buts domicile'] == résultats['Buts extérieur'], 'Résultat'] = 'N'
        résultats_finaux = résultats_finaux._append(résultats, ignore_index=True)

    return résultats_finaux


#on crée la liste des url et des noms de classes correspondants pour les saisons de Ligue 1 depuis 2010
liste = [
    ["https://www.footballcritic.com/ligue-1/season-2022-2023/matches/4/65453",67146],
    ["https://www.footballcritic.com/ligue-1/season-2021-2022/matches/4/51604",55330],
    ["https://www.footballcritic.com/ligue-1/season-2020-2021/matches/4/35278",46313],
    ["https://www.footballcritic.com/ligue-1/season-2019-2020/matches/4/21586",32460],
    ["https://www.footballcritic.com/ligue-1/season-2018-2019/matches/4/16810",22368],
    ["https://www.footballcritic.com/ligue-1/season-2017-2018/matches/4/13295",12876],
    ["https://www.footballcritic.com/ligue-1/season-2016-2017/matches/4/11798",8078],
    ["https://www.footballcritic.com/ligue-1/season-2015-2016/matches/4/7793",5880],
    ["https://www.footballcritic.com/ligue-1/season-2014-2015/matches/4/3817",3643],
    ["https://www.footballcritic.com/ligue-1/season-2013-2014/matches/4/2730",2751],
    ["https://www.footballcritic.com/ligue-1/season-2012-2013/matches/4/166",170],
    ["https://www.footballcritic.com/ligue-1/season-2011-2012/matches/4/167",171],
    ["https://www.footballcritic.com/ligue-1/season-2010-2011/matches/4/168",172],
    ["https://www.footballcritic.com/ligue-1/season-2009-2010/matches/4/169",173],
    ["https://www.footballcritic.com/ligue-1/season-2008-2009/matches/4/170",174],
    ["https://www.footballcritic.com/ligue-1/season-2007-2008/matches/4/171",175],
    ["https://www.footballcritic.com/ligue-1/season-2006-2007/matches/4/172",176],
    ["https://www.footballcritic.com/ligue-1/season-2005-2006/matches/4/173",177],
    ["https://www.footballcritic.com/ligue-1/season-2004-2005/matches/4/174",178],
    ["https://www.footballcritic.com/ligue-1/season-2003-2004/matches/4/175",179],
    ["https://www.footballcritic.com/ligue-1/season-2002-2003/matches/4/176",180]
]

#on créé le data frame final en parcourant les éléments de la liste et en utilisant notre fonction 
noms_colonnes = ['Saison','Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
dataframe_résultats = pd.DataFrame(columns=noms_colonnes)
for element in liste:
    try:
        résultats_finaux = fonction_résultats(element[0],element[1])
        dataframe_résultats = dataframe_résultats._append(résultats_finaux, ignore_index=True)
    except Exception as e:
        print(f"Erreur à l'élément {element}: {e}")

dataframe_résultats.to_csv('dataframe_résultats.csv', index=False)


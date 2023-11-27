#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

#on crée une fonction qui permettent de scrapper les résultats de chaque journée, en prenant en entrée un url et un nom de classe HTML
#on définit la fonction
def fonction_resultats(url,classe):
    #on récupère le script HTML grace à BeautifulSoup
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")
    #on crée un data frame
    noms_colonnes = ['Saison','Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
    resultats_finaux = pd.DataFrame(columns=noms_colonnes)
    #on isole dans le l'url la saison dont il est question
    url_parts = url.split("/")
    annee_part = [part for part in url_parts if "season" in part][0]
    annee = annee_part.replace("season-", "")
    #en raison du Covid, le nombre de journée n'est pas le même en 2019-2020
    if annee == "2019-2020":
        x=29
    else:
        x=39
    
    #on crée une boucle pour avoir chacune des 38 journées
    for i in range(1,x):
        #on isole dans la page le tableau contenant les résultats
        tableau = soup.find_all("span", class_="rounds r_" + str(classe) + " w_" + str(i))
        #on récupère la liste des équipes
        equipes = []
        for element in tableau:
            equipes.append(element.select('.info-box .text.hidden-xs a'))
        #on met au propre la liste des équipes
        equipes_finales =[]
        for equipe in equipes:
            for balise_a in equipe:
                texte_equipe = BeautifulSoup(str(balise_a), 'html.parser').get_text()
                equipes_finales.append(texte_equipe)
        equipes_finales = [equipe.strip() for equipe in equipes_finales]

        #dans notre liste on a [équipe, adversaire, équipe, adversaire ...]
        #on peut donc récupérer la liste des équipes jouant à domicile et celle des équipes jouant à l'extérieur
        equipes_domicile = equipes_finales[::2]
        equipes_exterieur = equipes_finales[1::2]

        #on récupère la liste des scores
        scores = []
        for element in tableau:
            scores.append(element.select('.btn-info'))
        #on met au propre la liste des scores
        scores_int =[]
        for element in scores:
            #en raison de certains matchs qui ne se sont pas joué (Strasbourg-PSG 2019-2020) ou où le résultat a été changé après le match (Nantes-Bastia 2013-2014), il faut être vigilant 
            #dans ces cas là au lieu du score, il y a soit l'heure du match soit un tiret
            #dans les années étudiées, on trouve 4 de ces occurences
            scores_match = re.findall(r'>-<|\d+\s*-\s*\d+|>20:45<|>21:00<|>17:00<',str(element))
            scores_int.extend(scores_match) 
        #on décide de mettre 0-0 pour les 4 matches en question 
        scores_int = ['0 - 0' if item in ('>-<','>20:45<','>21:00<','>17:00<') else item for item in scores_int]  
        #on met au propre la liste des scores
        scores_int = [element.split(' - ') for element in scores_int]
        scores_finaux = [item for sublist in scores_int for item in sublist]

        #on récupère la liste des scores des équipes jouant à domicile et celle des scores des équipes jouant à l'extérieur   
        scores_domicile = scores_finaux[::2]
        scores_exterieur = scores_finaux[1::2]

        #on peut maintenant compléter notre dataframe
        resultats = pd.DataFrame(columns=noms_colonnes)
        resultats['Saison']=[annee]*10
        resultats['Journée']= [i]*10
        resultats['Domicile'] = equipes_domicile
        resultats['Extérieur'] = equipes_exterieur
        resultats['Buts domicile'] = scores_domicile
        resultats['Buts extérieur'] = scores_exterieur
        #on met le résultat du match ("D" pour victoire de Domicile, "N" pour Nul et "E" pour victoire d'Extérieur)
        resultats['Résultat'] = -1
        resultats.loc[resultats['Buts domicile'] > resultats['Buts extérieur'], 'Résultat'] = 1
        resultats.loc[resultats['Buts domicile'] == resultats['Buts extérieur'], 'Résultat'] = 0
        #on ajoute les données pour chaques journées
        resultats_finaux = resultats_finaux._append(resultats, ignore_index=True)
    
    #on retourne le data frame final des résultats de la saison
    return resultats_finaux


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
dataframe_resultats = pd.DataFrame(columns=noms_colonnes)
#on ajoute les résultats pour chaque saison, en vérifiant qu'il n'y a pas d'erreur
for element in liste:
    try:
        resultats_finaux = fonction_resultats(element[0],element[1])
        dataframe_resultats = dataframe_resultats._append(resultats_finaux, ignore_index=True)
    except Exception as e:
        print(f"Erreur à l'élément {element}: {e}")
    
#on enregistre le dataframe final au format csv
dataframe_resultats.to_csv('dataframe_résultats.csv', encoding = 'utf-8', index=False)


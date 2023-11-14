#on importe les bibliothèques utiles
import pandas as pd
from bs4 import BeautifulSoup
import requests

#on cherche à scrapper les données de classement à chaque journée de la saison 2022/2023
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2022-2023"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

#on isole le tableau qui nous intéresse
classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[1:]:
    equipes.append(balise.text)

#on crée un dataframe#
colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_22_23 = pd.DataFrame(columns=colonnes)

#on y ajoute les classements pour chaque journée
x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+41*i:39+41*i]]
    liste = [equipes[i]] + liste
    classement_22_23.loc[len(classement_22_23)] = liste

#on fait la même chose pour la saison 2021/2022
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2021-2022"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[5:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_21_22 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
#il y a une coquille dans le script html de wiki (manque une colonne sur la ligne de l'OGC Nice) donc on découpe le code en deux parties
for i in range(5):
    liste = [x.get_text(strip=True) for x in x[1+42*i:39+42*i]]
    liste = [equipes[i]] + liste
    classement_21_22.loc[len(classement_21_22)] = liste
for i in range(5,20):
    liste = [x.get_text(strip=True) for x in x[0+42*i:38+42*i]]
    liste = [equipes[i]] + liste
    classement_21_22.loc[len(classement_21_22)] = liste


#on fait la même chose pour la saison 2020/2021
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2020-2021"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[7:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_20_21 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+42*i:39+42*i]]
    liste = [equipes[i]] + liste
    classement_20_21.loc[len(classement_20_21)] = liste


#on fait la même chose pour la saison 2019/2020. Année COVID oblige, certains petits changements s'imposent.
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2019-2020"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[6:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 29)]
classement_19_20 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+32*i:29+32*i]]
    liste = [equipes[i]] + liste
    classement_19_20.loc[len(classement_19_20)] = liste

#on fait la même chose pour la saison 2018/2019
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2018-2019"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[7:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_18_19 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+42*i:39+42*i]]
    liste = [equipes[i]] + liste
    classement_18_19.loc[len(classement_18_19)] = liste

#on fait la même chose pour la saison 2017/2018
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2017-2018"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[4:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_17_18 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+42*i:39+42*i]]
    liste = [equipes[i]] + liste
    classement_17_18.loc[len(classement_17_18)] = liste

#on fait la même chose pour la saison 2016/2017
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2016-2017"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[8:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_16_17 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+42*i:39+42*i]]
    liste = [equipes[i]] + liste
    classement_16_17.loc[len(classement_16_17)] = liste

#on fait la même chose pour la saison 2015/2016.
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2015-2016"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

#Le nom de la classe ici n'est pas le même que précédemment
classement = soup.find("table", class_="wikitable sortable center nowrap")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[0:]:
    equipes.append(balise.text)

colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
classement_15_16 = pd.DataFrame(columns=colonnes)

x = classement.find_all('td')
for i in range(20):
    liste = [x.get_text(strip=True) for x in x[1+41*i:39+41*i]]
    liste = [equipes[i]] + liste
    classement_15_16.loc[len(classement_15_16)] = liste

#Ici le classement n'est pas en ordre croissant, on rajoute donc une ligne de code
classement_15_16["J38"] = classement_15_16["J38"].astype(int)
classement_15_16 = classement_15_16.sort_values(by="J38")
classement_15_16 = classement_15_16.reset_index(drop=True)
print(classement_15_16)
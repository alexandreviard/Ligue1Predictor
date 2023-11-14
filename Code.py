#on importe les bibliothèques utiles#
import pandas as pd
from bs4 import BeautifulSoup
import requests

#on cherche à scrapper les données de classement à chaque journée#
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2022-2023"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")

#on isole le tableau qui nous intéresse"
classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[1:]:
    equipes.append(balise.text)

#on crée un dataframe#
colonnes = ["Équipes"] + ["J{}".format(i) for i in range(1, 39)]
df = pd.DataFrame(columns=colonnes)

for i in range(20):
    x = classement.find_all('td')
    liste = [x.get_text(strip=True) for x in x[1+41*i:39+41*i]]
    liste = [equipes[i]] + liste
    df.loc[len(df)] = liste
print(df)

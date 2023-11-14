import pandas as pd
from bs4 import BeautifulSoup
import requests
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_football_2022-2023"
page = requests.get(url)
soup = BeautifulSoup(page.text, "html.parser")
classement = soup.find("table", class_="wikitable sortable center")
equipes =[]
balises_a = classement.find_all('a')
for balise in balises_a[1:]:
    equipes.append(balise.text)

data = {'équipes': equipes}
for i in range(1, 39):
    nom_colonne = 'J{}'.format(i)
    data[nom_colonne] = [''] * len(equipes)
df = pd.DataFrame(data)
print(df)
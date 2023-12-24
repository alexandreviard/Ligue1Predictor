import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_ligue1_data(nb_saisons=6):
    """
    Cette fonction scrape les données de Ligue 1 pour un nombre donné de saisons.

    :param nb_saisons: Nombre de saisons de données à scraper.
    :return: DataFrame contenant les données agrégées de chaque saison.
    """
    # URL de base pour les données de la Ligue 1
    url_base_ligue1 = "https://fbref.com/en/comps/13/Ligue-1-Stats"

    # En-tête de l'utilisateur pour simuler une requête depuis un navigateur
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0'
    }


    all_seasons_data = []  # Liste pour stocker les données de toutes les saisons

    for saison in range(nb_saisons):
        time.sleep(3)  # Applique le contrôle du taux de requêtes

        # Scraping des URLs des équipes pour la saison courante
        response = requests.get(url_base_ligue1, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        teams_urls = [f"https://fbref.com{a.get('href')}" for a in soup.select("table.stats_table a") if "squads" in a.get('href')]

        # Mise à jour de l'URL pour la prochaine saison
        url_base_ligue1 = f"https://fbref.com{soup.find('a', class_='button2 prev').get('href')}"

        for team_url in teams_urls:
            time.sleep(3)  # Applique le contrôle du taux de requêtes

            # Scraping des données de l'équipe
            team_response = requests.get(team_url, headers=headers)
            team_data = pd.read_html(team_response.text, match="Scores")[0]
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
            team_data["Team"] = team_name

            # Scraping des URLs des statistiques détaillées
            soup_team = BeautifulSoup(team_response.text, 'html.parser')
            stats_urls = [f"https://fbref.com{a.get('href')}" for a in soup_team.find_all("a") if "matchlogs/all_comps" in a.get('href', '') and any(x in a.get('href', '') for x in ["passing/", "shooting", "possession/", "defense/", "keeper"])]

            for stats_url in set(stats_urls):
                time.sleep(3) # Applique le contrôle du taux de requêtes

                # Scraping des statistiques détaillées
                stats_response = requests.get(stats_url, headers=headers)
                detailed_stats = pd.read_html(stats_response.text)[0]
                if detailed_stats.columns.nlevels > 1:
                    detailed_stats.columns = [f"{col}_{branch}" if "For" not in col and "Unnamed:" not in col else f"{branch}" for col, branch in detailed_stats.columns]

                columns_to_drop = ["Time", "Comp", "Round", "Day", "Venue", "Result", "GF", "GA", "Opponent"] + [col for col in detailed_stats.columns if 'Report' in col]
                detailed_stats.drop(columns_to_drop, axis=1, inplace=True)

                team_data = team_data.merge(detailed_stats, on="Date")

            all_seasons_data.append(team_data)

    # Concaténation des données de toutes les saisons
    final_data = pd.concat(all_seasons_data, ignore_index=True)
    return final_data

# Exemple d'utilisation de la fonction
df_ligue1 = scrape_ligue1_data(nb_saisons=6)


def scrap_ligue1_last():
    url_ligue1 = "https://fbref.com/en/comps/13/Ligue-1-Stats"
left_on = ["Time", "Comp", "Round", "Day", "Venue", "Result", "GF", "GA", "Opponent"]
df_final = []

# En-tête de l'utilisateur pour ressembler à Mozilla Firefox
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0'
}


request1 = requests.get(url_ligue1, headers=headers)
soup = BeautifulSoup(request1.text, 'html.parser')
url_equipes = soup.select("table.stats_table")[0].find_all("a")
url_equipes = [equipes.get("href") for equipes in url_equipes]
url_equipes = [equipes for equipes in url_equipes if equipes and "squads" in equipes]
url_equipes = [f"https://fbref.com{i}" for i in url_equipes]



for j in range(len(url_equipes)):

    time.sleep(3)  # Applique le contrôle du taux de requêtes

    request2 = requests.get(url_equipes[j], headers=headers)
    soup2 = BeautifulSoup(request2.text, 'html.parser')
    df_equipe = pd.read_html(request2.text, match="Scores")[0]

    nom_equipe = url_equipes[j].split("/")[-1].replace("-Stats", "").replace("-", " ")
    df_equipe["equipe"] = nom_equipe

    url_stats = soup2.findAll("a")
    url_stats = [el.get("href") for el in url_stats]
    url_stats = [el for el in url_stats if el and "matchlogs/all_comps" in el]
    url_stats = [el for el in url_stats if el and ("passing/" in el or "shooting" in el or "possession/" in el or "defense/" in el or "keeper" in el)]
    url_stats = list(set(url_stats))
    url_stats = [f"https://fbref.com{i}" for i in url_stats]


    for y in range(len(url_stats)):
        time.sleep(3)  # Applique le contrôle du taux de requêtes

        request3 = requests.get(url_stats[y], headers=headers)
        soup3 = BeautifulSoup(request3.text, 'html.parser')
        stats = pd.read_html(request3.text)[0]

        if stats.columns.nlevels > 1:
            stats.columns = [f"{col}_{branch}" if "For" not in col and "Unnamed:" not in col else f"{branch}" for col, branch in stats.columns]

        stats.drop(left_on + [col for col in stats.columns if 'Report' in col], axis=1, inplace=True)

        df_equipe = df_equipe.merge(stats, on="Date")

    df_final.append(df_equipe)

df_final2 = pd.concat(df_final)


def add_new_matches(base_initiale, base_nouvelle):

    # Le scrapping change l'odre des colonnes il faut donc les réaligner pour concaténer
    base_nouvelle = base_nouvelle[base_initiale.columns]

    # Concaténation des deux bases de données et suppression des doublons
    concatenated_data = pd.concat([base_nouvelle, base_initiale]).drop_duplicates().sort_values(by ="DateTime").reset_index(drop=True)

    return concatenated_data
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time


# Fonction pour scraper les données de la Ligue 1 sur plusieurs saisons
def scrape_ligue1_data(nb_saisons=6):
    """
    Scrappe les données de la Ligue 1 pour un nombre donné de saisons.
    :param nb_saisons: Nombre de saisons à scraper.
    :return: DataFrame des données agrégées sur les saisons spécifiées.
    """
    url_base_ligue1 = "https://fbref.com/en/comps/13/Ligue-1-Stats"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_seasons_data = []

    for saison in range(nb_saisons):
        rate_limit()
        response = requests.get(url_base_ligue1, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        teams_urls = [f"https://fbref.com{a.get('href')}" for a in soup.select("table.stats_table a") if "squads" in a.get('href')]
        url_base_ligue1 = f"https://fbref.com{soup.find('a', class_='button2 prev').get('href')}"

        for team_url in teams_urls:
            rate_limit()
            team_response = requests.get(team_url, headers=headers)
            team_data = pd.read_html(team_response.text, match="Scores")[0]
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
            team_data["Team"] = team_name
            stats_urls = get_stats_urls(team_response)

            for stats_url in set(stats_urls):
                rate_limit()
                detailed_stats = get_detailed_stats(stats_url, headers)
                team_data = team_data.merge(detailed_stats, on="Date")

            all_seasons_data.append(team_data)

    return pd.concat(all_seasons_data, ignore_index=True)


# Fonction pour scraper les dernières données de la Ligue 1
def scrape_latest_ligue1_data():
    """
    Scrappe les dernières données disponibles pour chaque équipe de la Ligue 1.
    :return: DataFrame des dernières données agrégées pour chaque équipe.
    """
    url_ligue1 = "https://fbref.com/en/comps/13/Ligue-1-Stats"
    headers = {'User-Agent': 'Mozilla/5.0'}
    latest_data = []

    response = requests.get(url_ligue1, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    teams_urls = [f"https://fbref.com{a.get('href')}" for a in soup.select("table.stats_table a") if "squads" in a.get('href')]

    for team_url in teams_urls:
        rate_limit()
        team_response = requests.get(team_url, headers=headers)
        team_data = pd.read_html(team_response.text, match="Scores")[0]
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        team_data["Team"] = team_name
        stats_urls = get_stats_urls(team_response)

        for stats_url in set(stats_urls):
            rate_limit()
            detailed_stats = get_detailed_stats(stats_url, headers)
            team_data = team_data.merge(detailed_stats, on="Date")

        latest_data.append(team_data)

    return pd.concat(latest_data, ignore_index=True)


# Fonction pour limiter la fréquence des requêtes
def rate_limit():
    """
    Limite la fréquence des requêtes pour éviter de surcharger le serveur.
    """
    MIN_REQUEST_INTERVAL = 3
    last_request_time = getattr(rate_limit, "last_request_time", None)
    if last_request_time is not None:
        elapsed_time = time.time() - last_request_time
        if elapsed_time < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed_time)
    rate_limit.last_request_time = time.time()


# Fonctions auxiliaires pour obtenir les URLs des statistiques détaillées et les statistiques elles-mêmes
def get_stats_urls(team_response):
    soup_team = BeautifulSoup(team_response.text, 'html.parser')
    return [f"https://fbref.com{a.get('href')}" for a in soup_team.find_all("a") if "matchlogs/all_comps" in a.get('href', '') and any(x in a.get('href', '') for x in ["passing/", "shooting", "possession/", "defense/", "keeper"])]


def get_detailed_stats(stats_url, headers):
    stats_response = requests.get(stats_url, headers=headers)
    detailed_stats = pd.read_html(stats_response.text)[0]
    if detailed_stats.columns.nlevels > 1:
        detailed_stats.columns = [f"{col}_{branch}" if "For" not in col and "Unnamed:" not in col else f"{branch}" for col, branch in detailed_stats.columns]
    columns_to_drop = ["Time", "Comp", "Round", "Day", "Venue", "Result", "GF", "GA", "Opponent"] + [col for col in detailed_stats.columns if 'Report' in col]
    detailed_stats.drop(columns_to_drop, axis=1, inplace=True)
    return detailed_stats


# Fonction pour fusionner les données de base avec les nouvelles données
def add_new_matches(base_initiale, base_nouvelle):
    """
    Concatène les données de base avec les nouvelles données tout en éliminant les doublons.
    :param base_initiale: DataFrame de base.
    :param base_nouvelle: Nouveau DataFrame à ajouter.
    :return: DataFrame concaténé.
    """
    base_nouvelle = base_nouvelle[base_initiale.columns]
    return pd.concat([base_nouvelle, base_initiale]).drop_duplicates().sort_values(by="DateTime").reset_index(drop=True)


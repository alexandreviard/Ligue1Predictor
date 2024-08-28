import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_ligue1_data(nb_saisons=7):
    """
    Récupère les données de la Ligue 1 sur plusieurs saisons spécifiées.
    
    Args:
        nb_saisons (int): Nombre de saisons à récupérer.

    Returns:
        DataFrame: Un DataFrame Pandas contenant les données agrégées.
    """
    # Définition de l'URL de base et des en-têtes pour les requêtes
    url_base_ligue1 = "https://fbref.com/en/comps/13/Ligue-1-Stats"
    headers = {'User-Agent': 'Mozilla/5.0'}
    all_seasons_data = []

    # Boucle sur le nombre de saisons demandé
    for saison in range(nb_saisons):
        # Application du contrôle de taux de requête
        rate_limit()

        # Exécution de la requête et analyse HTML
        response = requests.get(url_base_ligue1, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Récupération des URLs des équipes
        teams_urls = ["https://fbref.com" + equipe.get("href") 
                      for equipe in soup.select("table.stats_table")[0].find_all("a") 
                      if "squads" in equipe.get("href", "")]

        # Mise à jour de l'URL pour la prochaine saison
        url_base_ligue1 = f"https://fbref.com{soup.find('a', class_='button2 prev').get('href')}"

        # Traitement pour chaque équipe
        for team_url in teams_urls:
            # Application du contrôle de taux de requête
            rate_limit()

            # Récupération et préparation des données de l'équipe
            team_response = requests.get(team_url, headers=headers)
            team_data = pd.read_html(team_response.text, match="Scores")[0]
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
            team_data["Team"] = team_name

            # Récupération des URLs pour les statistiques détaillées
            stats_urls = get_stats_urls(team_response)

            # Traitement des statistiques détaillées
            for stats_url in set(stats_urls):
                rate_limit()
                detailed_stats = get_detailed_stats(stats_url, headers)
                team_data = team_data.merge(detailed_stats, on="Date")

            # Ajout des données de l'équipe au résultat global
            all_seasons_data.append(team_data)

    # Concaténation de toutes les données des saisons
    return pd.concat(all_seasons_data, ignore_index=True)


def scrape_latest_ligue1_data():
    """
    Récupère les dernières données disponibles pour chaque équipe de la Ligue 1.

    Returns:
        2 DataFrame:
         - Un DataFrame Pandas avec les dernières données de chaque équipe.
         - Un DataFrame qui contient les futurs journées à venir à processer

    """
    # Configuration initiale similaire à scrape_ligue1_data()
    url_ligue1 = "https://fbref.com/en/comps/13/Ligue-1-Stats"
    headers = {'User-Agent': 'Mozilla/5.0'}
    latest_data = []
    futur_matchweek = []

    response = requests.get(url_ligue1, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    teams_urls = ["https://fbref.com" + equipe.get("href") 
                  for equipe in soup.select("table.stats_table")[0].find_all("a") 
                  if "squads" in equipe.get("href", "")]

    # Traitement similaire à scrape_ligue1_data() pour chaque équipe
    for team_url in teams_urls:
        rate_limit()
        team_response = requests.get(team_url, headers=headers)
        team_data = pd.read_html(team_response.text, match="Scores")[0]
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        team_data["Team"] = team_name

        futur_matchweek.append(team_data)
        stats_urls = get_stats_urls(team_response)

        for stats_url in set(stats_urls):
            rate_limit()
            print(team_url, stats_urls)
            detailed_stats = get_detailed_stats(stats_url, headers)
            team_data = team_data.merge(detailed_stats, on="Date")

        latest_data.append(team_data)

    # Retourne les données concaténées de toutes les équipes
    return pd.concat(latest_data, ignore_index=True), pd.concat(futur_matchweek, ignore_index=True)


def rate_limit():
    """
    Fonction pour limiter le taux de requêtes et éviter de surcharger le serveur.
    Utilise un délai d'attente entre les requêtes pour respecter les limitations.
    """
    MIN_REQUEST_INTERVAL = 3
    last_request_time = getattr(rate_limit, "last_request_time", None)
    if last_request_time is not None:
        elapsed_time = time.time() - last_request_time
        if elapsed_time < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed_time)
    rate_limit.last_request_time = time.time()


def get_stats_urls(team_response):
    """
    Récupère les URLs pour les statistiques détaillées d'une équipe.

    Args:
        team_response: Réponse de la requête de la page de l'équipe.

    Returns:
        Set[str]: Ensemble d'URLs des statistiques détaillées.
    """
    soup_team = BeautifulSoup(team_response.text, 'html.parser')

    # Extraction des URLs pour les statistiques détaillées
    url_stats = {
        f"https://fbref.com{a.get('href')}" 
        for a in soup_team.find_all("a") 
        if "matchlogs/all_comps" in a.get('href', '') and 
           any(substring in a.get('href', '') for substring in ["passing/", "shooting", "possession/", "defense/", "keeper"])
    }
    return url_stats


def get_detailed_stats(stats_url, headers):
    """
    Récupère les statistiques détaillées à partir de l'URL donnée.

    Args:
        stats_url: URL de la page des statistiques détaillées.
        headers: En-têtes pour la requête.

    Returns:
        DataFrame: DataFrame des statistiques détaillées.
    """
    stats_response = requests.get(stats_url, headers=headers)
    detailed_stats = pd.read_html(stats_response.text)[0]

    # Nettoyage des colonnes du DataFrame
    if detailed_stats.columns.nlevels > 1:
        detailed_stats.columns = [f"{col}_{branch}" 
                                  if "For" not in col and "Unnamed:" not in col 
                                  else f"{branch}" 
                                  for col, branch in detailed_stats.columns]
    columns_to_drop = ["Time", "Comp", "Round", "Day", "Venue", "Result", "GF", "GA", "Opponent"] + [col for col in detailed_stats.columns if 'Report' in col]
    detailed_stats.drop(columns_to_drop, axis=1, inplace=True)

    return detailed_stats


def add_new_matches(base_initiale, base_nouvelle):
    """
    Ajoute les nouvelles données à la base de données initiale en évitant les doublons.

    Args:
        base_initiale: DataFrame de la base de données initiale.
        base_nouvelle: DataFrame des nouvelles données à ajouter.

    Returns:
        DataFrame: DataFrame mis à jour avec les nouvelles données.
    """

    # Suppression de la colonne "Unnamed: 0" si elle existe
    if 'Unnamed: 0' in base_initiale.columns:
        base_initiale.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in base_nouvelle.columns:
        base_nouvelle.drop(columns=['Unnamed: 0'], inplace=True)

    # Concaténation de la base initiale et de la base nouvelle
    concatenated_df = pd.concat([base_initiale, base_nouvelle])
    
    # Suppression des doublons basée sur les colonnes "Date", "Team" et "Opponent"
    concatenated_df_dedup = concatenated_df.drop_duplicates(subset=['Date', 'Team', 'Opponent'])
    
    # Tri par "Date" et "Team"
    concatenated_df_sorted = concatenated_df_dedup.sort_values(by=['Date', 'Team'])

    # Réinitialisation de l'index
    concatenated_df_final = concatenated_df_sorted.reset_index(drop=True)
    
    return concatenated_df_final

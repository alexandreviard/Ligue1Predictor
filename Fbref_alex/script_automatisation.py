from scrapping import *
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from fonctions_preprocess import *
from sklearn.ensemble import RandomForestClassifier

mapping_equipe = {
        'Nimes': 'Nîmes',
        'Paris S-G': 'Paris Saint Germain',
        'Saint Etienne': 'Saint-Étienne'
    }

def update_data():

    base_actuelle = pd.read_csv("/home/onyxia/work/Projet-python/Fbref_alex/SOCCER_241223.csv")
    data = scrape_latest_ligue1_data()
    base_supplémentaire = data[0]

    base_updated = add_new_matches(base_initiale=base_actuelle, base_nouvelle=base_supplémentaire)

    base_updated.to_csv("/home/onyxia/work/Projet-python/Fbref_alex/SOCCER_241223.csv")

    recup_matchweek = data[1]

    recup_matchweek.to_csv("/home/onyxia/work/Projet-python/Fbref_alex/recup_matchweek.csv")

    return

def automatisation():

    a = pd.read_csv("/home/onyxia/work/Projet-python/Fbref_alex/recup_matchweek.csv", index_col=0)
    b = pd.read_csv("/home/onyxia/work/Projet-python/Fbref_alex/SOCCER_241223.csv", index_col=0)

    c = preparation_modelisation(base_initial=b, matchweeks=a, mapping_equipe= mapping_equipe)
    c = preparation_model(c)

    # Créer des variables factices pour la colonne 'Opponent'
    dummies = pd.get_dummies(c['Opponent'])

    # Fusionner les variables factices avec le DataFrame original
    c = pd.concat([c, dummies], axis=1)



    # Séparer les données en ensembles d'entraînement et de test
    train_c = c[c['Result'].notna()]
    test_c = c[c['Result'].isna()]


    # Filtrer pour conserver uniquement les colonnes avec "_Lag1" et les colonnes des variables factices
    features = [col for col in train_c.columns if '_Lag1' in col] + list(dummies.columns)


    # Préparer les ensembles d'entraînement et de test
    X_train = train_c[features]
    y_train = train_c['Result']
    X_test = test_c[features]

    # Supprimer les lignes avec des NaN dans les colonnes de features dans l'ensemble d'entraînement
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]  # Assurez-vous que y_train a les mêmes lignes que X_train

    # Créer et entraîner le modèle Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)

    # Ajouter les prédictions à l'ensemble de test
    test_c['Predicted_Result'] = predictions


    test_c = affichage_colonne_stockage(test_c)

    test_c.to_csv("Resultats.csv")

    return







def find_futur_matchweeks(df, mapping_equipe):
    
    # Supprimer les lignes où les colonnes 'Date', 'Time' et 'Round' sont manquantes.
    df.dropna(subset=["Date", "Time", "Round"], inplace=True)

    # Prétraiter le DataFrame en utilisant la fonction 'preprocess_intitial' et le mapping des équipes.
    df = preprocess_intitial(df, mapping_equipe)

    # Obtenir la date et l'heure actuelles.
    ajd = datetime.now()

    # Filtrer pour garder seulement les matchs programmés après la date et l'heure actuelles.
    df = df[df['DateTime'] >= ajd]

    # Trier le DataFrame en fonction de la colonne 'DateTime' dans l'ordre croissant.
    df = df.sort_values(by='DateTime')

    # Si le DataFrame n'est pas vide, obtenir la date du premier match à venir.
    # Sinon, définir 'premiere_date_proche' à None.
    premiere_date_proche = df['DateTime'].iloc[0] if not df.empty else None

    # Calculer la date qui est 10 jours après la 'premiere_date_proche'.
    dix_jours = timedelta(days=10) + premiere_date_proche

    # Filtrer pour garder seulement les matchs programmés dans les 10 jours suivant la 'premiere_date_proche'.
    df = df[df['DateTime'] <= dix_jours]

    return df


def preparation_modelisation(matchweeks, base_initial, mapping_equipe):

    a = find_futur_matchweeks(matchweeks, mapping_equipe)

    b = preprocess_intitial(base_initial, mapping_equipe)

    b = preprocess_variables(b)

    resultat = pd.concat([a, b], sort=False)

    return resultat





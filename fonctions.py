import pandas as pd
import time
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import numpy as np
from datetime import datetime, timedelta
import openpyxl
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from IPython.display import display
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder  



#Fonctions pour la deuxième approche:

def preprocess_initial(df, mapping_equipe):
    """
    Cette fonction effectue plusieurs opérations de prétraitement sur nos données footballistiques scrappées.
    
    :param df: DataFrame contenant les données footballistiques.
    :param mapping_equipe: Dictionnaire pour la normalisation des noms des équipes.
    """

    # Conversion et nettoyage des colonnes 'Date' et 'Time' en une seule colonne
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    elif all(col in df.columns for col in ["Date", "Time"]):
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.drop(["Date", "Time"], axis=1, inplace=True)
        df = df[['DateTime'] + [col for col in df.columns if col != 'DateTime']]

    # Normalisation des noms des équipes pour qu'il n'y est pas de noms de mêmes équipes différentes
    df['Opponent'] = df['Opponent'].map(mapping_equipe).fillna(df['Opponent'])
    df['Team'] = df['Team'].map(mapping_equipe).fillna(df['Team'])

    # Garder que les matchs 'Ligue 1' (pas de matchs de Coupe)
    df = df[df["Comp"] == "Ligue 1"]

    # Extraire uniquement le numéro de chaque journée (en ligue 1 il y'a 38 journées par an, ici on ne garde que le numéro)
    try:
        df['Round'] = df['Round'].str.extract(r'(\d+)').astype(int)
    except AttributeError:
        pass  # La colonne 'Round' n'était pas de type chaîne de caractères, aucune modification nécessaire
    except KeyError:
        pass  # La colonne 'Round' n'existe pas, pas de modification nécessaire

    # Création d'une colonne "Saison" qui nous permettra de facilement accéder au matchs d'une année partiulière
    df['Saison'] = df['DateTime'].apply(lambda x: f"{x.year}-{x.year + 1}" if x.month >= 8 else f"{x.year - 1}-{x.year}")
    
    # Nettoyage de la colonne 'Formation'
    if 'Formation' in df.columns:
        # Vérifie si la colonne 'Formation' est présente dans le DataFrame.
        # Remplace le caractère '◆' par une chaîne vide dans la colonne 'Formation'.
        df['Formation'] = df['Formation'].apply(lambda x: x.replace('◆', '') if pd.notnull(x) else x)

    # Vérifiez si les deux colonnes existent
    if 'Poss_x' in df.columns and 'Poss_y' in df.columns:
        # Renommer "Poss_x" en "Poss"
        df = df.rename(columns={'Poss_x': 'Poss'})

        # Supprimer la colonne "Poss_y"
        df = df.drop(columns=['Poss_y'])
    
    df['MatchID'] = df['Team'] + '_' + df['Opponent']
        
    return df


def preprocess_variables(df):

    """
    Cette fonction effectue plusieurs opérations de créations de variables pertinentes
    
    :param df: DataFrame contenant les données footballistiques.
    :param mapping_equipe: Dictionnaire pour la normalisation des noms des équipes.
    """

    # Création de la variable de différence entre buts marqués et encaissés
    df[['GF', 'GA']] = df[['GF', 'GA']].astype(float).astype(int)
    df['GD'] = df['GF'] - df['GA']

    # Création d'une colonne "Points" pour chaque match joués
    df['Points'] = df['Result'].map({'W': 3, 'D': 1, 'L': 0})

    # Calcul cumulatif des Points, des buts marqués/encaissés et de la différence de buts par saison
    df.sort_values(by=['Saison', 'Round', 'Team'], inplace=True)
    df.reset_index(drop=True, inplace=True)


    cumulative_cols = df.groupby(['Saison', 'Team']).agg({
        'Points': 'cumsum',
        'GD': 'cumsum',
        'GF': 'cumsum',
        'GA': 'cumsum'
    }).reset_index()

    df[['Points_Cum', 'GD_Cum', 'GF_Cum', 'GA_Cum']] = cumulative_cols[['Points', 'GD', 'GF', 'GA']]


    # Calculer un classement basé sur les points cumulés et la différence de buts
    df.sort_values(by=['Saison', 'Round', 'Points_Cum', 'GD_Cum'], ascending=[True, True, False, False], inplace=True)
    df['Classement'] = df.groupby(['Saison', 'Round']).cumcount() + 1

    outcome_cols = ['IsWin', 'IsDraw', 'IsLoss']
    df['Past_Matches'] = df.groupby('MatchID').cumcount()
    df['IsWin'] = df['Result'].apply(lambda x: 1 if x == 'W' else 0)
    df['IsDraw'] = df['Result'].apply(lambda x: 1 if x == 'D' else 0)
    df['IsLoss'] = df['Result'].apply(lambda x: 1 if x == 'L' else 0)


    df[['CumulativeWins', 'CumulativeDraws', 'CumulativeLosses']] = df.groupby('MatchID')[outcome_cols].cumsum()

    df["Total_Goals"] = df["GF"] + df["GA"]
    # Créer la colonne "Minus 1.5 Goals"
    df["Minus 1.5 Goals"] = (df["Total_Goals"] <= 1.5).astype(int)

    # Créer la colonne "Minus 2.5 Goals"
    df["Minus 2.5 Goals"] = (df["Total_Goals"] <= 2.5).astype(int)

    # Créer la colonne "Minus 3.5 Goals"
    df["Minus 3.5 Goals"] = (df["Total_Goals"] <= 3.5).astype(int)



    # Réinitialisation de l'index et tri final
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['Saison', 'Team', 'DateTime'], inplace=True)

    return df



def preparation_model(df):
    """
    Prépare les données pour la modélisation en ajoutant des variables dérivées :
    - Variables lagged (décalées) : Points, différence de buts, buts pour et contre
    - Classement et statistiques cumulatives des dernières rencontres entre équipes
    - Moyennes mobiles décalées pour diverses statistiques de match
    """

    # Tri initial par saison, round et équipe
    df.sort_values(by=['Saison', 'Round', 'Team'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Création de variables décalées pour les statistiques cumulatives
    lag_cols = ['Points_Cum', 'GD_Cum', 'GF_Cum', 'GA_Cum']
    df[[f'{col}_Lag1' for col in lag_cols]] = df.groupby(['Saison', 'Team'])[lag_cols].shift(1)

    # Décalage du classement pour chaque équipe
    df['Classement_Lag1'] = df.groupby(['Team'])['Classement'].shift(1)

    # Création d'un indicateur de forme sur les derniers matchs
    df['FormeW_Lag'] = df.groupby(['Saison', 'Team'])['IsWin'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=5).sum())
    df['FormeL_Lag'] = df.groupby(['Saison', 'Team'])['IsLoss'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=5).sum())

    # Tri par points et différence de buts cumulés
    df.sort_values(by=['Saison', 'Round', 'Points_Cum', 'GD_Cum'], ascending=[True, True, False, False], inplace=True)

    # Décalage des statistiques de victoires, nuls et défaites
    df[['CumulativeWins_Lag1', 'CumulativeDraws_Lag1', 'CumulativeLosses_Lag1']] = df.groupby('MatchID')[['CumulativeWins', 'CumulativeDraws', 'CumulativeLosses']].shift(1)



    # Liste des colonnes de statistiques pour calculer les moyennes mobiles décalées
    stat_columns = [
        "Total Shots", "Shots on Target", "Shots on Target %", "Goals per Shot", "Total Touches", 
        "Touches in Defensive Penalty Area", "Touches in Defensive Third", "Touches in Midfield Third", 
        "Touches in Attacking Third", "Touches in Attacking Penalty Area", "Dribbles Attempted", 
        "Successful Dribbles", "Successful Dribble %", "Total Carries", "Total Carry Distance", 
        "Progressive Carry Distance", "Progressive Carries", "Carries into Final Third", 
        "Carries into Penalty Area", "Tackles", "Tackles Won", "Tackles in Defensive Third", 
        "Tackles in Midfield Third", "Tackles in Attacking Third", "Dribblers Tackled", 
        "Total Dribbles Against", "Defensive Dribblers Win %", "Interceptions", "Errors Leading to Goal", 
        "Key Passes", "Passes Completed", "Passes Attempted", "Passes into Final Third", 
        "Progressive Passes", "Shots on Target Against", "Keeper Saves", "Keeper Save Percentage"]

    for col in stat_columns:
        df[f'Moyenne_{col}_Lag'] = df.groupby(['Saison', 'Team'])[col].transform(lambda x: x.shift(1).expanding().mean())

    # Réorganisation finale par saison, équipe et date
    df = df.sort_values(by=['Saison', 'Team', 'DateTime']).reset_index(drop=True)

    return df


def renommer_colonnes(df):
    
    precise_renaming_dict = {
        
        # Standard stats

        'Standard_Sh': 'Total Shots',
        'Standard_SoT': 'Shots on Target',
        'Standard_SoT%': 'Shots on Target %',
        'Standard_G/Sh': 'Goals per Shot',
        'Standard_G/SoT': 'Goals per Shot on Target',
        'Standard_Dist': 'Average Shot Distance',
        'Standard_FK': 'Free Kicks Taken',
        'Standard_PK': 'Penalty Kicks Scored',
  
        # Expected stats
        'Expected_xG': 'Expected Goals',
        'Expected_npxG': 'Non-Penalty Expected Goals',
        'Expected_npxG/Sh': 'Non-Penalty Expected Goals per Shot',
        'Expected_G-xG': 'Goal Difference vs Expected Goals',
        'Expected_np:G-xG': 'Non-Penalty Goal Difference vs Expected Goals',

        # Touches
        'Touches_Touches': 'Total Touches',
        'Touches_Def Pen': 'Touches in Defensive Penalty Area',
        'Touches_Def 3rd': 'Touches in Defensive Third',
        'Touches_Mid 3rd': 'Touches in Midfield Third',
        'Touches_Att 3rd': 'Touches in Attacking Third',
        'Touches_Att Pen': 'Touches in Attacking Penalty Area',

        # Take-Ons (Dribbles)
        'Take-Ons_Att': 'Dribbles Attempted',
        'Take-Ons_Succ': 'Successful Dribbles',
        'Take-Ons_Succ%': 'Successful Dribble %',
        'Take-Ons_Tkld': 'Dribbles Tackled',
        'Take-Ons_Tkld%': 'Dribble Tackle %',

        # Carries = "Contrôle de balle au pied"
        'Carries_Carries': 'Total Carries',
        'Carries_TotDist': 'Total Carry Distance',
        'Carries_PrgDist': 'Progressive Carry Distance', #towards opponent goal
        'Carries_PrgC': 'Progressive Carries', #10 yards from its furthest point towards opponent goal
        'Carries_1/3': 'Carries into Final Third',
        'Carries_CPA': 'Carries into Penalty Area',
        'Carries_Mis': 'Carries Miscontrolled',
        'Carries_Dis': 'Carries Dispossessed',


        # Defensive Actions
        'Tackles_Tkl': 'Tackles',
        'Tackles_TklW': 'Tackles Won',
        'Tackles_Def 3rd': 'Tackles in Defensive Third',
        'Tackles_Mid 3rd': 'Tackles in Midfield Third',
        'Tackles_Att 3rd': 'Tackles in Attacking Third',
        'Challenges_Tkl': 'Dribblers Tackled',
        'Challenges_Att': 'Total Dribbles Against',
        'Challenges_Tkl%': 'Defensive Dribblers Win %',
        'Challenges_Lost': 'Defensive Challenges Lost',
        'Blocks_Blocks': 'Total Blocks',
        'Blocks_Sh': 'Shot Blocks',
        'Blocks_Pass': 'Pass Blocks',
        'Int': 'Interceptions',
        'Tkl+Int': 'Tackles Plus Interceptions',
        'Clr': 'Clearances',
        'Err': 'Errors Leading to Goal',

        # Passing
        'Total_Cmp': 'Passes Completed',
        'Total_Att': 'Passes Attempted',
        'Total_Cmp%': 'Pass Completion %',
        'Total_TotDist': 'Total Pass Distance',
        'Total_PrgDist': 'Progressive Pass Distance', #towards opponent goal
        'Short_Cmp': 'Short Passes Completed',
        'Short_Att': 'Short Passes Attempted',
        'Short_Cmp%': 'Short Pass Completion %',
        'Medium_Cmp': 'Medium Passes Completed',
        'Medium_Att': 'Medium Passes Attempted',
        'Medium_Cmp%': 'Medium Pass Completion %',
        'Long_Cmp': 'Long Passes Completed',
        'Long_Att': 'Long Passes Attempted',
        'Long_Cmp%': 'Long Pass Completion %',

        # Creative Play
        'Ast': 'Assists',
        'xAG': 'Expected Assists Goals',
        'xA': 'Expected Assists',
        'KP': 'Key Passes',
        '1/3': 'Passes into Final Third',
        'PPA': 'Passes into Penalty Area',
        'CrsPA': 'Crosses into Penalty Area',
        'PrgP': 'Progressive Passes', #pass that move forward from the furthest point (at leats 10 yards)

        # Goalkeeping
        'Performance_SoTA': 'Shots on Target Against',
        'Performance_Saves': 'Keeper Saves',
        'Performance_Save%': 'Keeper Save Percentage',
        'Performance_CS': 'Clean Sheets', #0 ou 1

        # Penalty Kicks
        'Penalty Kicks_PKA': 'Penalty Kicks Against',
        'Penalty Kicks_PKsv': 'Penalty Kicks Against Saved',
        'Penalty Kicks_PKm': 'Penalty Kicks Against Missed',}

    # Renommer les colonnes selon le dictionnaire, en tenant compte de celles qui existent
    for old_col, new_col in precise_renaming_dict.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)


    # Liste des colonnes à supprimer
    columns_to_drop = [
        "Standard_Gls",
        "Launched_Cmp", 
        "Launched_Att", 
        "Launched_Cmp%", 
        "Passes_Att (GK)", 
        "Passes_Thr", 
        "Passes_Launch%", 
        "Passes_AvgLen", 
        "Goal Kicks_Att", 
        "Goal Kicks_Launch%", 
        "Goal Kicks_AvgLen", 
        "Crosses_Opp", 
        "Crosses_Stp", 
        "Crosses_Stp%", 
        "Sweeper_#OPA", 
        "Sweeper_AvgDist",
        "Penalty Kicks_PKatt",
        "Performance_GA",
        "Performance_PSxG",
        "Performance_PSxG+/-",
        "Receiving_Rec",
        "Touches_Live",
        "Standard_PKatt",
        "Receiving_PrgR"
        # Ajoutez d'autres noms de colonnes ici si nécessaire
    ]
    
    # Supprimer les colonnes non nécessaires, en vérifiant leur existence
    columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=columns_to_drop_existing, inplace=True)

    return df


def columns_to_keep(df):
    # Liste des colonnes à garder
    columns_to_keep = [
        "DateTime", "Comp", "Round", "Day", "Venue", "Result", "GF", "GA", "Opponent","MatchID","Saison", 
        "Attendance", "Captain", "Formation", "Referee", "Match Report", "Notes", "Team", 
        "Total Shots", "Shots on Target", "Shots on Target %", "Goals per Shot", "Total Touches", 
        "Touches in Defensive Penalty Area", "Touches in Defensive Third", "Touches in Midfield Third", 
        "Touches in Attacking Third", "Touches in Attacking Penalty Area", "Dribbles Attempted", 
        "Successful Dribbles", "Successful Dribble %", "Total Carries", "Total Carry Distance", 
        "Progressive Carry Distance", "Progressive Carries", "Carries into Final Third", 
        "Carries into Penalty Area", "Tackles", "Tackles Won", "Tackles in Defensive Third", 
        "Tackles in Midfield Third", "Tackles in Attacking Third", "Dribblers Tackled", 
        "Total Dribbles Against", "Defensive Dribblers Win %", "Interceptions", "Errors Leading to Goal", 
        "Key Passes", "Passes Completed", "Passes Attempted", "Passes into Final Third", 
        "Progressive Passes", "Shots on Target Against", "Keeper Saves", "Keeper Save Percentage"
    ]
    
    # Créer une liste des colonnes à garder qui existent réellement dans le DataFrame
    columns_to_keep_existing = [col for col in columns_to_keep if col in df.columns]
    
    # Sélectionner uniquement les colonnes existantes dans le DataFrame
    df = df[columns_to_keep_existing]

    return df


def preprocess_data(df):

    # Fonction pour créer le MatchID
    def create_match_id(row):
        teams = sorted([row['Team'], row['Opponent']])
        return f"{row['DateTime']}-{teams[0]}-vs-{teams[1]}"
    
    # Ajout de la colonne MatchID
    df['MatchID'] = df.apply(create_match_id, axis=1)

    # Colonnes qui doivent rester inchangées (sans suffixes)
    colonnes_fixes = ['DateTime', 'Comp', 'Round', 'Day', 'MatchID', 'Saison', 'Referee', 'Match Report', 'Notes', "Minus 1.5 Goals", "Minus 2.5 Goals", "Minus 3.5 Goals"]

    # Identification des colonnes variables (en excluant les colonnes fixes)
    colonnes_variables = [col for col in df.columns if col not in colonnes_fixes]

    # Séparation du dataframe en domicile et à l'extérieur
    df_domicile = df[df['Venue'] == 'Home'].copy()
    df_exterieur = df[df['Venue'] == 'Away'].copy()

    # Ajout de suffixes aux colonnes variables
    df_domicile = df_domicile.rename(columns={col: col + '_Home' for col in colonnes_variables})
    df_exterieur = df_exterieur.rename(columns={col: col + '_Away' for col in colonnes_variables})


    # Définition de la fonction determine_result
    def determine_result(row):
        if row['Result_Home'] == 'W':
            return 'W_Home'
        elif row['Result_Away'] == 'W':
            return 'W_Away'
        elif row['Result_Away'] == 'D':
            return 'D'
        else:
            return np.nan

    # Appliquer la fonction determine_result à chaque ligne de merged_df

    merged_df = pd.merge(df_domicile, df_exterieur, on=colonnes_fixes, how='inner')
    merged_df['Result'] = merged_df.apply(determine_result, axis=1)
    merged_df.rename(columns = {"Team_Home" : "Team Home", "Team_Away": "Team Away"}, inplace=True)

    return merged_df

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



def modelisation(df, cutoff_date, targets=["Result", "Minus 2.5 Goals"], model_type=None, plot_features=False):
    """
    Fonction pour créer et appliquer des modèles de prédiction pour plusieurs cibles.

    :param df: DataFrame contenant les données des matchs.
    :param cutoff_date: Date limite pour séparer les données d'entraînement et de test.
    :param targets: Liste des colonnes cibles pour la prédiction.
    :param plot_features: Booléen indiquant si les importances des caractéristiques doivent être tracées.
    :param model_type: Type de modèle de prédiction à utiliser. Si None, sélectionne automatiquement le meilleur modèle.
    :return: DataFrame avec les prédictions ajoutées pour chaque cible.
    """
    selected_columns = ["DateTime"] + [col for col in df.columns if col.endswith(('Lag1_Home', 'Lag1_Away', 'Lag_Home', 'Lag_Away'))]

    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1),
        'SVC': SVC(random_state=42, probability=True)
    }

    label_encoders = {target: LabelEncoder() for target in targets}

    for target in targets:
        
        X = df[selected_columns].copy()
        X[target] = df[target].astype('category')
        X_train = X[df['DateTime'] <= cutoff_date].dropna()
        y_train = X_train[target]
        X_train.drop(columns=[target, 'DateTime'], errors='ignore', inplace=True)
        y_train = label_encoders[target].fit_transform(y_train)

        
        if plot_features:
            X_test = X[df['DateTime'] > cutoff_date].drop(columns=['DateTime'], errors='ignore').dropna(subset=[col for col in selected_columns if col != 'DateTime'])
            y_test = X_test[target]
            X_test.drop(columns=target, inplace= True)
        else:
            X_test = X[df['DateTime'] > cutoff_date].drop(columns=[target,'DateTime'], errors='ignore').dropna(subset=[col for col in selected_columns if col != 'DateTime'])


        # Sélectionner le meilleur modèle via la validation croisée si aucun modèle n'est spécifié
        best_model = model_type

        if best_model is None:
            best_score = 0
            for name, model in models.items():

                score = np.mean(cross_val_score(model, X_train, y_train, cv=3))
                if score > best_score:
                    best_score = score
                    best_model = name

        selected_model = models[best_model]


        # Gestion du Suréchantillonnage
        oversampler = RandomOverSampler(sampling_strategy='all', random_state=5)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

        # Entraînement du modèle sélectionné
        selected_model.fit(X_train_resampled, y_train_resampled)


        # Prédiction des résultats et calcul des probabilités

        y_pred = selected_model.predict(X_test)
        df.loc[X_test.index, f'Predicted_{target}'] = label_encoders[target].inverse_transform(y_pred)


        if hasattr(selected_model, "predict_proba"):
            df.loc[X_test.index, f'Prediction_Probability_{target}'] = np.max(selected_model.predict_proba(X_test), axis=1)
        else:
            df.loc[X_test.index, f'Prediction_Probability_{target}'] = np.nan

        # Si plot_features est True et le modèle est un RandomForest, tracer les importances des caractéristiques
        if plot_features and isinstance(selected_model, RandomForestClassifier):
            importances = selected_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 6))
            plt.title(f'Importance des features: {target}')
            plt.bar(range(X_train_resampled.shape[1]), importances[indices], align='center')
            plt.xticks(range(X_train_resampled.shape[1]), X_train_resampled.columns[indices], rotation=90)
            plt.tight_layout()
            plt.show()

        if plot_features:
            # Calcul de l'accuracy
            accuracy = accuracy_score(label_encoders[target].transform(y_test.dropna()), y_pred[:len(y_test.dropna())])
            print(f"Précision sur les prédictions : '{target}' avec {best_model}: {accuracy:.2f}")



    # Colonnes à retourner
    return_cols = ["DateTime", "Comp", "Saison", "Round", "Day", "Team Home", "GF_Home", "GF_Away", "Team Away", "Result"] + \
                  [col for target in targets for col in (f'Predicted_{target}', f'Prediction_Probability_{target}')] +  ["MatchID"]

    return df[return_cols][(df['DateTime'] > cutoff_date) & (df['Predicted_Result'].notnull())]


"""
def modelisation(df, cutoff_date):
    
    targets = ["Result", "Minus 2.5 Goals"]
    selected_columns = ["DateTime"] + [col for col in df.columns if col.endswith(('Lag1_Home', 'Lag1_Away', 'Lag_Home', 'Lag_Away'))]

    # Boucle sur chaque cible
    for target in targets:
        X = df[selected_columns].copy()
        X[target] = df[target].astype('category')

        # Séparation des données
        X_train = X[df['DateTime'] <= cutoff_date].dropna()
        y_train = X_train[target]
        X_train.drop(columns=[target, 'DateTime'], errors='ignore', inplace=True)

        X_test = X[df['DateTime'] > cutoff_date].drop(columns=[target, 'DateTime'], errors='ignore').dropna(subset=[col for col in selected_columns if col != 'DateTime'])

        # Gestion du Suréchantillonnage des modalités
        oversampler = RandomOverSampler(sampling_strategy='all', random_state=5)
        X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

        # Modèle et entraînement (Random Forest avec paramètres par défaults)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_resampled, y_train_resampled)

        # Prédiction et probabilités
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        max_proba = np.max(y_pred_proba, axis=1)

        # Ajout des prédictions et probabilités
        df.loc[X_test.index, f'Predicted_{target}'] = y_pred
        df.loc[X_test.index, f'Prediction_Probability_{target}'] = max_proba

    # Colonnes à retourner
    return_cols = ["DateTime", "Comp", "Saison", "Round", "Day", "Team Home", "GF_Home", "GF_Away", "Team Away", "Result"] + \
                  [col for target in targets for col in (f'Predicted_{target}', f'Prediction_Probability_{target}')] +  ["MatchID"]

    
    return df[return_cols][(df['DateTime'] > cutoff_date) & (df['Predicted_Result'].notnull())]

"""


def find_futur_matchweeks(df, mapping_equipe):

    """
    À partir du scrapping on récupère un DataFrame qui contient les futurs journées, il faut le mettre en forme.
    args: DataFrame, le mapping ligue1
    """
    
    # Supprimer les lignes où les colonnes 'Date', 'Time' et 'Round' sont manquantes.
    df.dropna(subset=["Date", "Time", "Round"], inplace=True)

    # Prétraiter le DataFrame en utilisant la fonction 'preprocess_intitial' et le mapping des équipes.
    df = preprocess_initial(df, mapping_equipe)

    # Obtenir la date et l'heure actuelles.
    ajd = datetime.now()

    # Filtrer pour garder seulement les matchs programmés après la date et l'heure actuelles.
    df = df[df['DateTime'] >= ajd]

    # Trier le DataFrame en fonction de la colonne 'DateTime' dans l'ordre croissant.
    df = df.sort_values(by='DateTime')

    # Si le DataFrame n'est pas vide, obtenir la date du premier match à venir.
    # Sinon, définir 'premiere_date_proche' à None.
    if df.empty != True:
        premiere_date_proche = df['DateTime'].iloc[0]
    else:
        return None
    
    # Calculer la date qui est 10 jours après la 'premiere_date_proche'.
    dix_jours = timedelta(days=10) + premiere_date_proche

    # Filtrer pour garder seulement les matchs programmés dans les 10 jours suivant la 'premiere_date_proche'.
    df = df[df['DateTime'] <= dix_jours]

    # Si 'premiere_date_proche' est None, retourner None.
    if df.empty == True:
        return None
    else:# Retourner le DataFrame s'il n'est pas vide, sinon retourner None.
        return df

def main_process(data):
    data = renommer_colonnes(data)
    data = preprocess_initial(data, mapping_equipe)
    data = columns_to_keep(data)
    data = preprocess_variables(data)
    data = preparation_model(data)
    data = preparation_model(data)
    return data



# Fonctions pour la première approche

def fonction_resultats(i):
    url = 'https://fbref.com/en/comps/13/' + str(i) +'-' + str(i+1) + '/schedule/' + str(i) +'-' + str(i+1) + '-Ligue-1-Scores-and-Fixtures'
    page = requests.get(url)
    html_content = page.content
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if i == 2023:
        df = pd.read_html(str(table))[0].dropna(subset = 'Wk').reset_index(drop=True)
        df['Wk'] = df['Wk'].astype(int)
        affiches = df[df['Score'].isna()]
        affiches[['Home', 'Away']] = affiches[['Home', 'Away']].replace('Paris S-G', 'Paris Saint Germain')

    else:
        df = pd.read_html(str(table))[0].dropna(subset = 'Wk').dropna(subset = 'Score').reset_index(drop=True)
        affiches = pd.DataFrame()
    df = df[['Wk', 'Home', 'Score', 'Away']]
    noms_colonnes = ['Saison','Journée','Domicile','Extérieur','Buts domicile','Buts extérieur','Résultat']
    df.insert(0,'Saison', str(i) + '-' + str(i+1))
    df['Buts domicile'] = df['Score'].apply(lambda x: 0 if pd.isna(x) else int(x[0])).astype(int)
    df['Buts extérieur'] = df['Score'].apply(lambda x: 0 if pd.isna(x) else int(x[2])).astype(int)
    df.drop(['Score'], axis = 1, inplace = True)
    df['Résultat'] = -1
    df.loc[df['Buts domicile'] > df['Buts extérieur'], 'Résultat'] = 1
    df.loc[df['Buts domicile'] == df['Buts extérieur'], 'Résultat'] = 0
    df.columns = noms_colonnes
    df['Journée'] = df['Journée'].astype(int)
    df[['Domicile', 'Extérieur']] = df[['Domicile', 'Extérieur']].replace('Paris S-G', 'Paris Saint Germain')
    return df, affiches

def fonction_prepa_base (dataframe_final, i):
    dataframe_final.insert(4, 'Equipe 1 à Domicile', 1)
    noms_colonnes = ['Saison', 'Journée', 'Equipe 1', 'Equipe 2', 'Equipe 1 à Domicile', 'Buts Equipe 1', 'Buts Equipe 2', 'Résultat']
    dataframe_final.columns = noms_colonnes
    dataframe_final_copie = dataframe_final.copy()[['Saison', 'Journée', 'Equipe 2', 'Equipe 1', 'Equipe 1 à Domicile', 'Buts Equipe 2', 'Buts Equipe 1', 'Résultat']]
    dataframe_final_copie.columns = noms_colonnes
    dataframe_final_copie['Equipe 1 à Domicile'] = 0
    dataframe_final = dataframe_final._append(dataframe_final_copie, ignore_index=True)
    dataframe_final = dataframe_final.sort_values(by=['Saison', 'Equipe 1', 'Journée']).reset_index(drop=True)
    conditions = [
        (dataframe_final['Buts Equipe 1'] > dataframe_final['Buts Equipe 2']),
        (dataframe_final['Buts Equipe 1'] < dataframe_final['Buts Equipe 2'])
    ]
    valeurs = [1, -1]
    dataframe_final['Résultat'] = 0
    dataframe_final['Résultat'] = np.select(conditions, valeurs)

    dataframe_final['GF'] = dataframe_final['Buts Equipe 1'].astype(float).astype(int)
    dataframe_final['GA'] = dataframe_final['Buts Equipe 2'].astype(float).astype(int)

    dataframe_final['GD'] = dataframe_final['GF']- dataframe_final['GA']
    dataframe_final['Points'] = dataframe_final['Résultat'].map({1: 3, 0: 1, -1: 0})

    dataframe_final.sort_values(by=['Saison', 'Journée', 'Equipe 1']).reset_index()
    result = dataframe_final.groupby(['Saison', 'Equipe 1']).agg({
        'Points': 'cumsum',
        'GD': 'cumsum',
        'GF': 'cumsum',
        'GA' : 'cumsum'
    }).reset_index()

    dataframe_final[['CPoints', 'CGD', 'CGF', 'CGA']] = result[['Points', 'GD', 'GF', 'GA']]
    dataframe_final = dataframe_final.sort_values(by=['Saison', 'Journée', 'CPoints', 'CGD', 'CGF'], ascending=[True, True, False, False, False]).reset_index()
    dataframe_final['Classement Equipe 1'] = dataframe_final.groupby(['Saison', 'Journée']).cumcount() + 1
    dataframe_final['Classement Equipe 1'] = dataframe_final.groupby(['Saison','Equipe 1'])['Classement Equipe 1'].shift(i).astype(pd.Int64Dtype())
    dataframe_final = dataframe_final.sort_values(by=['Saison', 'Equipe 1', 'Journée']).reset_index(drop=True).drop(['index','Points', 'GD', 'GF', 'GA','CPoints', 'CGD', 'CGF', 'CGA'], axis=1)

    dataframe_final['Classement Equipe 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Classement Equipe 1_y']
    dataframe_final['Moyenne_BM par 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].transform(lambda x: x.shift(i).expanding().mean())
    dataframe_final['Moyenne_BM par 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BM par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum() - i*dataframe_final['Buts Equipe 1']) / 
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() +1-i)
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BM par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final['Moyenne_BM par 2 à Domicile'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 à Domicile_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BM par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum() - i*dataframe_final['Buts Equipe 1']) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() +1-i)
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BM par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final['Moyenne_BM par 2 à Extérieur'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 à Extérieur_y']
    dataframe_final['Moyenne_BE par 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].transform(lambda x: x.shift(i).expanding().mean())
    dataframe_final['Moyenne_BE par 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BE par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum() - i*dataframe_final['Buts Equipe 2']) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() + 1-i)
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BE par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final['Moyenne_BE par 2 à Domicile'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 à Domicile_y']
    
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BE par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum() - i*dataframe_final['Buts Equipe 2']) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount()+1-i)
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BE par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount()))
    dataframe_final['Moyenne_BE par 2 à Extérieur'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 à Extérieur_y']
    dataframe_final['Forme 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Résultat'].rolling(window=i+5, min_periods=i+1).sum().reset_index(drop=True) - i*dataframe_final['Résultat']
    dataframe_final['Forme 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Forme 1_y']
    dataframe_final['Historique'] = dataframe_final.groupby(['Equipe 1', 'Equipe 2'])['Résultat'].cumsum() - i*dataframe_final['Résultat']
    if i == 0:
        dataframe_final['Résultat'] = dataframe_final['Résultat'].map({1: 'Victoire', 0: 'Nul', -1: 'Défaite'})
        dataframe_final['Equipe 1 à Domicile'] = dataframe_final['Equipe 1 à Domicile'].map({1: 'Domicile', 0: 'Extérieur'})
        dataframe_final = dataframe_final.rename(columns={'Equipe 1 à Domicile': 'Lieu'})
    return dataframe_final



def fonction_appli_modeles(dataframe_regression, X):
    random_forest_model = RandomForestClassifier(n_estimators=1000, random_state=5)
    svm_model = svm.SVC(kernel='rbf', C=50, random_state=42)
    X['poids'] = 1 + (dataframe_regression['Journée'] - 1) * 0.1
    Y = dataframe_regression[["Résultat"]]
    Z = dataframe_regression[["Buts Equipe 1"]]
    W = dataframe_regression[["Buts Equipe 2"]]
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, W_train, W_test = train_test_split(X, Y, Z, W, test_size=0.2, random_state=42)
    weights_train = X_train['poids']
    X_train1 = X_train.drop(['poids'], axis=1)
    X_test1 = X_test.drop(['poids'], axis=1)  

    #Régression sur le résultat
    X_train1 = sm.add_constant(X_train1)
    X_test1 = sm.add_constant(X_test1)
    model = sm.WLS(Y_train.astype(float), X_train1.astype(float), weights=weights_train)
    results = model.fit()

    print(results.summary())

    Y_pred = results.predict(X_test1)
    Y_pred = [-1 if x < 0 else 1 for x in Y_pred]
    Y_test = Y_test[Y_test.columns[0]] .tolist()

    bon_résultat = [a == b for a, b in zip(Y_pred, Y_test)]

    accuracy = (sum(bon_résultat) / len(bon_résultat)) 
    print('Régression sur le Résultat')
    print('accuracy: ', accuracy)

    #Régression sur les Scores dont on déduit un Résultat
    model = sm.WLS(Z_train.astype(float), X_train1.astype(float), weights_train)
    results = model.fit()

    print(results.summary())

    Z_pred = results.predict(X_test1)
    Z_test = Z_test[Z_test.columns[0]] .tolist()

    model = sm.WLS(W_train.astype(float), X_train1.astype(float), weights = weights_train)
    results = model.fit()

    print(results.summary())

    W_pred = results.predict(X_test1)
    W_test = W_test[W_test.columns[0]] .tolist()

    resultat = [a - b for a, b in zip(Z_pred, W_pred)]
    Y_pred2 = [-1 if x < 0 else 1 for x in  resultat]
    bon_résultat = [a == b for a, b in zip(Y_pred2, Y_test)]

    accuracy = (sum(bon_résultat) / len(bon_résultat)) 
    print('Régression sur les Scores dont on déduit un Résultat')
    print('accuracy: ', accuracy)


    #Random Forest sur le résultat
    X_train['Classement Equipe 1'] = 1/X_train['Classement Equipe 1']     #On repasse les classements en mode classique pour les deux prochains modèles
    X_train['Classement Equipe 2'] = 1/X_train['Classement Equipe 2']
    X_test['Classement Equipe 1'] = 1/X_test['Classement Equipe 1']
    X_test['Classement Equipe 2'] = 1/X_test['Classement Equipe 2']
    random_forest_model.fit(X_train, Y_train)
    Y_pred3 = random_forest_model.predict(X_test)


    accuracy = accuracy_score(Y_test, Y_pred3)
    classification_report_result = classification_report(Y_test, Y_pred3)

    print('Random Forest sur le Résultat')
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report_result)

    #SVM model sur le résultat
    svm_model.fit(X_train, Y_train)

    Y_pred4 = svm_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred4)
    classification_report_result = classification_report(Y_test, Y_pred4)

    print('SVM model sur le Résultat')
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report_result)

    return Y_test, Y_pred, Y_pred2, Y_pred3, Y_pred4


def trouver_chemins_images_avec_mot_cle(dossier, mot_cle):
    chemins_images = []

    fichiers = os.listdir(dossier)

    for fichier in fichiers:
        chemin_fichier = os.path.join(dossier, fichier)

        if os.path.isfile(chemin_fichier) and fichier.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            
            if mot_cle.lower() in fichier.lower():
                chemins_images.append(chemin_fichier)
                print(f"Chemin de l'image trouvée : {chemin_fichier}")
    return chemins_images[0]

def fonction_tableau_stats(df, first, equipe, venue):
    if venue == 'Domicile':
        noms_colonnes = ['Classement', 'Moyenne buts marqués', 'Moyenne buts encaissés', 'Moyenne buts marqués à Domicile', 'Moyenne buts encaissés à Domicile', 'Forme']
        columns_to_round = ['Moyenne_BM par 1', 'Moyenne_BM par 1 à Domicile', 'Moyenne_BE par 1', 'Moyenne_BE par 1 à Domicile']
        df_match = df[((df['Equipe 1'] == equipe) & (df['Journée'] == first))][['Classement Equipe 1', 'Moyenne_BM par 1','Moyenne_BE par 1', 'Moyenne_BM par 1 à Domicile', 'Moyenne_BE par 1 à Domicile']].reset_index(drop=True)
    else:
        noms_colonnes = ['Classement', 'Moyenne buts marqués', 'Moyenne buts encaissés', 'Moyenne buts marqués à Extérieur',  'Moyenne buts encaissés à Extérieur', 'Forme']
        columns_to_round = ['Moyenne_BM par 1', 'Moyenne_BM par 1 à Extérieur', 'Moyenne_BE par 1', 'Moyenne_BE par 1 à Extérieur']
        df_match = df[((df['Equipe 1'] == equipe) & (df['Journée'] == first))][['Classement Equipe 1', 'Moyenne_BM par 1','Moyenne_BE par 1', 'Moyenne_BM par 1 à Extérieur', 'Moyenne_BE par 1 à Extérieur']].reset_index(drop=True)

    
    df_match[columns_to_round] = df_match[columns_to_round].round(2)
    df['Résultat'] = df['Résultat'].replace({-1: 'D', 0: 'N', 1: 'V'})
    df_match['Forme'] = ''.join(
        str(df[((df['Equipe 1'] == equipe) & (df['Journée'] == first - i))]['Résultat'].iloc[0])
        for i in range(1, 6)
)
    df_match.columns = noms_colonnes
    return df_match





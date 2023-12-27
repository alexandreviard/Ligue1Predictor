import pandas as pd
import time
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import numpy as np
from datetime import datetime, timedelta

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


    # Réinitialisation de l'index et tri final
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by=['Saison', 'Team', 'DateTime'], inplace=True)

    return df


def affichage_colonne(df):

    # Réorganisez les colonnes dans le DataFrame
    colonnes_a_afficher_en_premier = ["DateTime", "Comp", "Round", "Day", "Venue", "Team", "Classement",  
                                  "Formation", "Result", "GF", "GA", "Opponent", "Past_Matches", 
                                  "CumulativeWins", "CumulativeDraws", "CumulativeLosses", 
                                  "Attendance", "Captain", "Referee"]
    # Réorganisez les colonnes dans le DataFrame
    #df = df[colonnes_a_afficher_en_premier + [col for col in df.columns if col not in colonnes_a_afficher_en_premier]]
   
    nouvelles_colonnes = colonnes_a_afficher_en_premier + [col for col in df.columns if col not in colonnes_a_afficher_en_premier]
    df = df[nouvelles_colonnes]

    return df
    


def preparation_model(df):
    """
    Prépare les données pour la modélisation en ajoutant des variables dérivées :
    - Variables lagged (décalées) : Points, différence de buts, buts pour et contre
    - Classement et statistiques cumulatives des dernières rencontres entre équipes
    - Moyennes mobiles décalées pour diverses statistiques de match
    Ensuite, les données sont réorganisées et les colonnes non nécessaires sont supprimées pour éviter la fuite de données.
    """

    # Tri initial par saison, round et équipe
    df.sort_values(by=['Saison', 'Round', 'Team'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Création de variables décalées pour les statistiques cumulatives
    lag_cols = ['Points_Cum', 'GD_Cum', 'GF_Cum', 'GA_Cum']
    df[[f'{col}_Lag1' for col in lag_cols]] = df.groupby(['Saison', 'Team'])[lag_cols].shift(1)

    # Décalage du classement pour chaque équipe
    df['Classement_Lag1'] = df.groupby(['Team'])['Classement'].shift(1)

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

    # Suppression des colonnes initiales pour éviter les fuites de données
    df.drop(stat_columns + lag_cols + ['Classement', 'CumulativeWins', 'CumulativeDraws', 'CumulativeLosses'], axis=1, inplace=True, errors='ignore')

    # Réorganisation finale par saison, équipe et date
    df = df.sort_values(by=['Saison', 'Team', 'DateTime']).reset_index(drop=True)

    return df



def affichage_colonne_stockage(df):

    # Réorganisez les colonnes dans le DataFrame
    colonnes_a_afficher_en_premier = ["DateTime", "Comp", "Round", "Day", "Venue", "Team", "Classement_Lag1",  
                                  "Predicted_Result", "Opponent"]
    # Réorganisez les colonnes dans le DataFrame
    #df = df[colonnes_a_afficher_en_premier + [col for col in df.columns if col not in colonnes_a_afficher_en_premier]]
   
    nouvelles_colonnes = colonnes_a_afficher_en_premier + [col for col in df.columns if col not in colonnes_a_afficher_en_premier]
    df = df[nouvelles_colonnes]

    return df



def renommer_colonnes(df):
    
    precise_renaming_dict = {
        # General match information
        #'GF': 'Goals For',
        #'GA': 'Goals Against',
        
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
    colonnes_fixes = ['DateTime', 'Comp', 'Round', 'Day', 'MatchID', 'Saison', 'Referee', 'Match Report', 'Notes']

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







def modelisation(df, cutoff_date):
    
    selected_columns = ["Result", "DateTime"] + [col for col in df.columns if col.endswith(('Lag1_Home', 'Lag1_Away', 'Lag_Home', 'Lag_Away'))]

    X = df[selected_columns].copy() 
    X["Result"] = X['Result'].astype('category')

    # Séparation des données
    X_train = X[df['DateTime'] <= cutoff_date].dropna()
    y_train = X_train["Result"]
    X_train.drop(columns=['Result', 'DateTime'], errors='ignore', inplace=True)
    X_test = X[df['DateTime'] > cutoff_date].drop(columns=['Result', 'DateTime'], errors='ignore').dropna(subset=[col for col in df.columns if col.endswith(('Lag1_Home', 'Lag1_Away', 'Lag_Home', 'Lag_Away'))])


    # Suréchantillonnage pour équilibrer toutes les classes
    oversampler = RandomOverSampler(sampling_strategy='all', random_state=5)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

    # Création et entraînement du modèle
    base_rf = RandomForestClassifier(random_state=42)
    base_rf.fit(X_train_resampled, y_train_resampled)

    # Prédiction et probabilités sur l'ensemble de test
    y_pred = base_rf.predict(X_test)
    y_pred_proba = base_rf.predict_proba(X_test)
    max_proba = np.max(y_pred_proba, axis=1)

    # Ajouter les prédictions et les probabilités au DataFrame
    df.loc[X_test.index, 'Predicted_Result'] = y_pred
    df.loc[X_test.index, 'Prediction_Probability'] = max_proba

    return df[["DateTime", "Comp", "Saison", "Round", "Day","Team Home", "Team Away", "Result", 'Predicted_Result', "Prediction_Probability", "MatchID"]][df['Predicted_Result'].notnull()]



def find_futur_matchweeks(df, mapping_equipe):

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
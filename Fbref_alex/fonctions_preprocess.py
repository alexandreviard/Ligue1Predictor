import pandas as pd
import time
from datetime import datetime

def preprocess_intitial(df, mapping_equipe):
    """
    Cette fonction effectue plusieurs opérations de prétraitement sur nos données footballistiques scrappées.
    
    :param df: DataFrame contenant les données footballistiques.
    :param mapping_equipe: Dictionnaire pour la normalisation des noms des équipes.
    """

    # Conversion et nettoyage des colonnes 'Date' et 'Time' en une seule colonne
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.drop(["Date", "Time"], axis=1, inplace=True)
    df = df[['DateTime'] + [col for col in df if col != 'DateTime']]

    # Normalisation des noms des équipes pour qu'il n'y est pas de noms de mêmes équipes différentes
    df['Opponent'] = df['Opponent'].map(mapping_equipe).fillna(df['Opponent'])
    df['Team'] = df['Team'].map(mapping_equipe).fillna(df['Team'])

    # Garder que les matchs 'Ligue 1' (pas de matchs de Coupe)
    df = df[df["Comp"] == "Ligue 1"]

    # Extraire uniquement le numéro de chaque journée (en ligue 1 il y'a 38 journées par an, ici on ne garde que le numéro)
    df['Round'] = df['Round'].str.extract(r'(\d+)').astype(int)

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

    """ # Application des décalages (lags)
    df[[f'{col}_Lag1' for col in lag_cols]] = df.groupby(['Saison', 'Team'])[lag_cols].shift(1)


    # Calcul des moyennes roulantes
    for col in mean_cols:
        df[f'Moyenne_{col}'] = df.groupby(['Saison', 'Team'])[col].transform(lambda x: x.shift(1).expanding().mean())
    """

    # Création d'un identifiant unique pour analyser les dernières rencontres entre deux équipes

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


#mean_cols = ['Standard_SoT%', 'Total_Cmp%', 'Poss_x', 'Touches_Def Pen', 'Touches_Def 3rd']
#outcome_cols = ['IsWin', 'IsDraw', 'IsLoss']
#lag_cols = ['Points_Cum', 'GD_Cum', 'GF_Cum', 'GA_Cum']

def variables_pertinentes(df):
    {'Standard_Sh': 'Total_shots',
     'Standard_SoT': "Shots_on_target",
     'Standard_SoT%' : 'SoT%'
      'Standar'}


def preparation_model(df):

    """ Une fois le preprocess utilisé sur la base, on prépare la base pour modéliser de la prédiction
    pour cela on créer des variables laggés, des variables par saisons, et des variables dont on aurait accès avant un match
    """

    # 0. Préparation pour le calcul cumulatif
    df.sort_values(by=['Saison', 'Round', 'Team'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Application des décalages (lags) sur les variables principales

    lag_cols = ['Points_Cum', 'GD_Cum', 'GF_Cum', 'GA_Cum']
    df[[f'{col}_Lag1' for col in lag_cols]] = df.groupby(['Saison', 'Team'])[lag_cols].shift(1)
    
    # 2. Création du classement et des cumulatives laggés des dernières rencontres entre les deux équipes
    df['Classement_Lag1'] = df.groupby(['Team'])['Classement'].shift(1)

    # Trier le DataFrame
    df.sort_values(by=['Saison', 'Round', 'Points_Cum', 'GD_Cum'], ascending=[True, True, False, False], inplace=True)

    # Créer les variables de décalage
    df[['CumulativeWins_Lag1', 'CumulativeDraws_Lag1', 'CumulativeLosses_Lag1']] = df.groupby('MatchID')[['CumulativeWins', 'CumulativeDraws', 'CumulativeLosses']].shift(1)


    # 3. Liste des colonnes de statistiques pour lesquelles calculer les moyennes mobiles décalées
    stat_columns = [
        col for col in df.columns 
        if col.startswith(('Standard_', 'Expected_', 'Poss_', 'Touches_', 'Take-Ons_', 'Carries_', 
                       'Receiving_', 'Tackles_', 'Challenges_', 'Blocks_', 'Total_', 'Short_', 
                       'Medium_', 'Long_', 'Performance_', 'Penalty Kicks_', 'Launched_', 
                       'Passes_', 'Goal Kicks_', 'Crosses_', 'Sweeper_'))
    ]

    for col in stat_columns:
        df[f'Moyenne_{col}_Lag'] = df.groupby(['Saison', 'Team'])[col].transform(lambda x: x.shift(1).expanding().mean())

    # Suppression des colonnes initiales de statistiques pour éviter les fuites de données (data leakage)
    df.drop(stat_columns, axis=1, inplace=True, errors='ignore')

    
    # Réorganisation finale du DataFrame
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
        'Standard_Gls': 'Goals Scored',
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

    # Renommer les colonnes selon le dictionnaire
    df.rename(columns=precise_renaming_dict, inplace=True)

    # Liste des colonnes à supprimer
    columns_to_drop = [
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
    
    # Supprimer les colonnes non nécessaires
    df.drop(columns=columns_to_drop, inplace=True)

    return df

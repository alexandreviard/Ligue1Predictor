import pandas as pd
import numpy as np 
import openpyxl
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
import os
from IPython.display import display



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
    else:
        df = pd.read_html(str(table))[0].dropna(subset = 'Wk').dropna(subset = 'Score').reset_index(drop=True)
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
    return df, affiches

def fonction_prepa_base (dataframe_final):
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
    dataframe_final['Classement Equipe 1'] = dataframe_final.groupby(['Saison','Equipe 1'])['Classement Equipe 1'].shift(1).astype(pd.Int64Dtype())
    dataframe_final = dataframe_final.sort_values(by=['Saison', 'Equipe 1', 'Journée']).reset_index(drop=True).drop(['index','Points', 'GD', 'GF', 'GA','CPoints', 'CGD', 'CGF', 'CGA'], axis=1)

    dataframe_final['Classement Equipe 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Classement Equipe 1_y']
    dataframe_final['Moyenne_BM par 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].transform(lambda x: x.shift(1).expanding().mean())
    dataframe_final['Moyenne_BM par 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BM par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum() - dataframe_final['Buts Equipe 1']) / 
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BM par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final['Moyenne_BM par 2 à Domicile'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 à Domicile_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BM par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum() - dataframe_final['Buts Equipe 1']) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BM par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final['Moyenne_BM par 2 à Extérieur'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 à Extérieur_y']
    dataframe_final['Moyenne_BE par 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].transform(lambda x: x.shift(1).expanding().mean())
    dataframe_final['Moyenne_BE par 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BE par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum() - dataframe_final['Buts Equipe 2']) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BE par 1 à Domicile'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final['Moyenne_BE par 2 à Domicile'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 à Domicile_y']
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 0, 'Moyenne_BE par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum() - dataframe_final['Buts Equipe 2']) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount())
    )
    dataframe_final.loc[dataframe_final['Equipe 1 à Domicile'] == 1, 'Moyenne_BE par 1 à Extérieur'] = (
        (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].cumsum() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum()) 
        / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount() - dataframe_final.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount()))
    dataframe_final['Moyenne_BE par 2 à Extérieur'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 à Extérieur_y']
    dataframe_final['Forme 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Résultat'].rolling(window=6, min_periods=2).sum().reset_index(drop=True)-dataframe_final['Résultat']
    dataframe_final['Forme 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Forme 1_y']
    dataframe_final['Historique'] = dataframe_final.groupby(['Equipe 1', 'Equipe 2'])['Résultat'].cumsum() - dataframe_final['Résultat']

    return dataframe_final

def fonction_prepa_stats (dataframe_stats):
    dataframe_stats['GF'] = dataframe_stats['Buts Equipe 1'].astype(float).astype(int)
    dataframe_stats['GA'] = dataframe_stats['Buts Equipe 2'].astype(float).astype(int)

    dataframe_stats['GD'] = dataframe_stats['GA']
    dataframe_stats['Points'] = dataframe_stats['Résultat'].map({1: 3, 0: 1, -1: 0})

    dataframe_stats.sort_values(by=['Saison', 'Journée', 'Equipe 1']).reset_index()
    result = dataframe_stats.groupby(['Saison', 'Equipe 1']).agg({
        'Points': 'cumsum',
        'GD': 'cumsum',
        'GF': 'cumsum',
        'GA' : 'cumsum'
    }).reset_index()

    dataframe_stats[['CPoints', 'CGD', 'CGF','CGA']] = result[['Points', 'GD', 'GF', 'GA']]
    dataframe_stats = dataframe_stats.sort_values(by=['Saison', 'Journée', 'CPoints', 'CGD', 'CGF'], ascending=[True, True, False, False, False]).reset_index()
    dataframe_stats['Classement Equipe 1'] = dataframe_stats.groupby(['Saison', 'Journée']).cumcount() + 1
    dataframe_stats['Classement Equipe 1'] = dataframe_stats.groupby(['Saison','Equipe 1'])['Classement Equipe 1'].shift(0).astype(pd.Int64Dtype())
    dataframe_stats = dataframe_stats.sort_values(by=['Saison', 'Equipe 1', 'Journée']).reset_index(drop=True).drop(['index','Points', 'GD', 'GF', 'GA','CPoints', 'CGD', 'CGF', 'CGA'], axis=1)

    dataframe_stats['Classement Equipe 2'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Classement Equipe 1_y']
    dataframe_stats['Moyenne_BM par 1'] = dataframe_stats.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].transform(lambda x: x.expanding().mean())
    dataframe_stats['Moyenne_BM par 2'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1_y']
    dataframe_stats.loc[dataframe_stats['Equipe 1 à Domicile'] == 1, 'Moyenne_BM par 1 à Domicile'] = (
        (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum()) / 
        (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() + 1)
    )

    dataframe_stats['Moyenne_BM par 2 à Domicile'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 à Domicile_y']
    dataframe_stats.loc[dataframe_stats['Equipe 1 à Domicile'] == 0, 'Moyenne_BM par 1 à Extérieur'] = (
        (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 1'].cumsum()) 
        / (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() + 1)
    )
    dataframe_stats['Moyenne_BM par 2 à Extérieur'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 à Extérieur_y']
    dataframe_stats['Moyenne_BE par 1'] = dataframe_stats.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].transform(lambda x: x.expanding().mean())
    dataframe_stats['Moyenne_BE par 2'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1_y']
    dataframe_stats.loc[dataframe_stats['Equipe 1 à Domicile'] == 1, 'Moyenne_BE par 1 à Domicile'] = (
        (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum()) 
        / (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() + 1)
    )
    dataframe_stats['Moyenne_BE par 2 à Domicile'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 à Domicile_y']
    dataframe_stats.loc[dataframe_stats['Equipe 1 à Domicile'] == 0, 'Moyenne_BE par 1 à Extérieur'] = (
        (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Buts Equipe 2'].cumsum()) 
        / (dataframe_stats.groupby(['Saison', 'Equipe 1', 'Equipe 1 à Domicile'])['Journée'].cumcount() + 1)
    )
    dataframe_stats['Moyenne_BE par 2 à Extérieur'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 à Extérieur_y']
    dataframe_stats['Forme 1'] = dataframe_stats.groupby(['Saison', 'Equipe 1'])['Résultat'].rolling(window=6, min_periods=1).sum().reset_index(drop=True)
    dataframe_stats['Forme 2'] = dataframe_stats.merge(dataframe_stats, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Forme 1_y']
    dataframe_stats['Historique'] = dataframe_stats.groupby(['Equipe 1', 'Equipe 2'])['Résultat'].cumsum()

    dataframe_stats['Résultat'] = dataframe_stats['Résultat'].map({1: 'Victoire', 0: 'Nul', -1: 'Défaite'})
    dataframe_stats['Equipe 1 à Domicile'] = dataframe_stats['Equipe 1 à Domicile'].map({1: 'Domicile', 0: 'Extérieur'})
    dataframe_stats = dataframe_stats.rename(columns={'Equipe 1 à Domicile': 'Lieu'})
    dataframe_stats = dataframe_stats.fillna(0)

    return dataframe_stats

def fonction_pred(df_train):
    dataframe_regression = df_train.dropna().copy()

    dataframe_regression['Classement Equipe 1'] = 1/dataframe_regression['Classement Equipe 1']
    dataframe_regression['Classement Equipe 2'] = 1/dataframe_regression['Classement Equipe 2']


    X_train = dataframe_regression.drop(['Equipe 1', 'Equipe 2', 'Saison', 'Journée', 'Buts Equipe 1', 'Buts Equipe 2', 'Résultat'], axis=1)
    X_train['poids'] = np.where(dataframe_regression['Journée'] > 15, 2, 1)
    weights_train = X_train['poids']
    X_train = X_train.drop(['poids'], axis=1)
    X_train1 = sm.add_constant(X_train)
    Y_train = dataframe_regression["Buts Equipe 1"]
    Y_train2 = dataframe_regression["Résultat"]


    random_forest_model2 = RandomForestClassifier(n_estimators=1000, random_state=5)
    random_forest_model2.fit(X_train, Y_train)

    model1 = sm.WLS(Y_train.astype(float), X_train1.astype(float), weights = weights_train)
    results1 = model1.fit()

    model2 = sm.WLS(Y_train2.astype(float), X_train1.astype(float), weights = weights_train)
    results2 = model2.fit()

    
    return random_forest_model2, model1, model2, 


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
        noms_colonnes = ['Classement', 'Moyenne buts marqués', 'Moyenne buts marqués à Domicile', 'Moyenne buts encaissés', 'Moyenne buts encaissés à Domicile', 'Forme']
        columns_to_round = ['Moyenne_BM par 1', 'Moyenne_BM par 1 à Domicile', 'Moyenne_BE par 1', 'Moyenne_BE par 1 à Domicile']
        df_match = df[((df['Equipe 1'] == equipe) & (df['Journée'] == first))][['Classement Equipe 1', 'Moyenne_BM par 1','Moyenne_BE par 1', 'Moyenne_BM par 1 à Domicile', 'Moyenne_BE par 1 à Domicile']].reset_index(drop=True)
    else:
        noms_colonnes = ['Classement', 'Moyenne buts marqués', 'Moyenne buts marqués à Extérieur', 'Moyenne buts encaissés', 'Moyenne buts encaissés à Extérieur', 'Forme']
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





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





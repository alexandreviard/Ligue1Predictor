import pandas as pd
import numpy as np 
import openpyxl
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 1000)
pd.options.display.float_format = '{:.2f}'.format    
dataframe_final = pd.read_csv('C:\\Users\\vtgra\\Desktop\\Projet python\\dataframe_résultats.csv',encoding = 'utf-8')
dataframe_classement = pd.read_csv('C:\\Users\\vtgra\\Desktop\\Projet python\\dataframe_classements.csv',encoding = 'utf-8')


glossaire = {'AS Monaco FC' : 'Monaco', 'FC Girondins de Bordeaux' : 'Bordeaux', 'FC Nantes Atlantique' : 'Nantes',
'Havre AC' : 'Le Havre', 'Montpellier Hérault SC' : 'Montpellier', 'Paris Saint-Germain FC' : 'PSG',
 'EA Guingamp' : 'Guingamp', 'Olympique Lyonnais' : 'Lyon', 'AC Ajaccio' : 'Ajaccio', 'SC Bastia' : 'Bastia', 'RC Lens' : 'Lens',
 'RC Strasbourg' : 'Strasbourg', 'CS Sedan Ardennes' : 'Sedan Ardennes', 'FC Sochaux-Montbéliard' : 'Sochaux', 'OGC Nice' : 'Nice',
 'AJ Auxerre' : 'Auxerre', 'Stade Rennais FC' : 'Rennais', 'Olympique de Marseille' : 'Marseille', 'Lille OSC' : 'Lille',
 'ESTAC Troyes' : 'Troyes', 'Toulouse FC' : 'Toulouse', 'Le Mans UC 72' : 'Le Mans', 'FC Metz' : 'Metz', 'SM Caen' : 'Caen',
 'FC Istres' : 'Istres', 'AS Saint-Etienne' : 'Saint-Étienne', 'AS Nancy Lorraine' : 'Nancy', 'FC Lorient' : 'Lorient',
 'Valenciennes FC' : 'Valenciennes', 'Grenoble Foot 38' : 'Grenoble', 'FC Nantes' : 'Nantes', 'AJA' : 'Auxerre' , 'FCGB':'Bordeaux', 'USB' : 'Boulogne',
 'GF38' : 'Grenoble', 'MUC' : 'Le Mans', 'RCL' : 'Lens', 'LOSC' : 'Lille', 'FCL' : 'Lorient', 'OL' : 'Lyon', 'OM' : 'Marseille', 'ASM' : 'Monaco', 
 'MHSC' : 'Montpellier', 'ASNL' : 'Nancy', 'OGCN' : 'Nice', 'SRFC' : 'Rennais', 'ASSE' : 'Saint-Étienne','FCSM' : 'Sochaux', 'TFC' : 'Toulouse', 
 'VAFC' : 'Valenciennes', 'ACAA' : 'Arles-Avignon', 'SB29' : 'Brest', 'SMC' : 'Caen', 'ACA' : 'Ajaccio', 'DFCO' : 'Dijon', 'ETGFC' : 'Thonon Évian',
 'Évian TG' : 'Thonon Évian', 'Paris' : 'PSG',  'Paris SG' : 'PSG', 'Nîmes' : 'Nimes', 'St-Étienne' : 'Saint-Étienne', 'Rennes' : 'Rennais'}

dataframe_classement['Équipes'] = dataframe_classement['Équipes'].replace(glossaire)

df_merge1 = pd.merge(dataframe_final, dataframe_classement, left_on=['Saison', 'Domicile'], right_on=['Saison', 'Équipes'], how='left')
conditions = [df_merge1['Journée'] == (i + 1) for i in range(1, max(df_merge1['Journée']) + 1)]
valeurs = [df_merge1[f'J{i}'] for i in range(1, max(df_merge1['Journée']) + 1)]
dataframe_final['Classement D'] = np.select(conditions, valeurs)
dataframe_final['Classement D'] = dataframe_final['Classement D'].replace({0: np.nan}).astype(pd.Int64Dtype())


df_merge2 = pd.merge(dataframe_final, dataframe_classement, left_on=['Saison', 'Extérieur'], right_on=['Saison', 'Équipes'], how='left')
conditions = [df_merge2['Journée'] == (i + 1) for i in range(1, max(df_merge2['Journée']) + 1)]
valeurs = [df_merge2[f'J{i}'] for i in range(1, max(df_merge2['Journée']) + 1)]
dataframe_final['Classement E'] = np.select(conditions, valeurs)
dataframe_final['Classement E'] = dataframe_final['Classement E'].replace({0: np.nan}).astype(pd.Int64Dtype())

dataframe_final.insert(4, 'Lieu', 'Domicile')
noms_colonnes = ['Saison', 'Journée', 'Equipe 1', 'Equipe 2', 'Lieu', 'Buts Equipe 1', 'Buts Equipe 2', 'Résultat', 'Classement Equipe 1', 'Classement Equipe 2']
dataframe_final.columns = noms_colonnes
dataframe_final_copie = dataframe_final.copy()[['Saison', 'Journée', 'Equipe 2', 'Equipe 1', 'Lieu', 'Buts Equipe 2', 'Buts Equipe 1', 'Résultat', 'Classement Equipe 2', 'Classement Equipe 1']]
noms_colonnes = ['Saison', 'Journée', 'Equipe 1', 'Equipe 2', 'Lieu', 'Buts Equipe 1', 'Buts Equipe 2', 'Résultat', 'Classement Equipe 1', 'Classement Equipe 2']
dataframe_final_copie.columns = noms_colonnes
dataframe_final_copie['Lieu'] = 'Extérieur'
dataframe_final = dataframe_final._append(dataframe_final_copie, ignore_index=True)
dataframe_final = dataframe_final.sort_values(by=['Saison', 'Equipe 1', 'Journée']).reset_index(drop=True)
conditions = [
    (dataframe_final['Buts Equipe 1'] > dataframe_final['Buts Equipe 2']),
    (dataframe_final['Buts Equipe 1'] < dataframe_final['Buts Equipe 2'])
]
valeurs = [1, -1]
dataframe_final['Résultat'] = 0
dataframe_final['Résultat'] = np.select(conditions, valeurs)


dataframe_final['Moyenne_BM par 1'] = (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 1'].cumsum() - dataframe_final['Buts Equipe 1']) / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount())
dataframe_final['Moyenne_BM par 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1_y']
dataframe_final['Moyenne_BM par 1 selon Lieu'] = (dataframe_final.groupby(['Saison', 'Equipe 1', 'Lieu'])['Buts Equipe 1'].cumsum() - dataframe_final['Buts Equipe 1']) / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Lieu'])['Journée'].cumcount())
dataframe_final['Moyenne_BM par 2 selon Lieu'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BM par 1 selon Lieu_y']
dataframe_final['Moyenne_BE par 1'] = (dataframe_final.groupby(['Saison', 'Equipe 1'])['Buts Equipe 2'].cumsum() - dataframe_final['Buts Equipe 2']) / (dataframe_final.groupby(['Saison', 'Equipe 1'])['Journée'].cumcount())
dataframe_final['Moyenne_BE par 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1_y']
dataframe_final['Moyenne_BE par 1 selon Lieu'] = (dataframe_final.groupby(['Saison', 'Equipe 1', 'Lieu'])['Buts Equipe 2'].cumsum() - dataframe_final['Buts Equipe 2']) / (dataframe_final.groupby(['Saison', 'Equipe 1', 'Lieu'])['Journée'].cumcount())
dataframe_final['Moyenne_BE par 2 selon Lieu'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Moyenne_BE par 1 selon Lieu_y']
dataframe_final['Forme 1'] = dataframe_final.groupby(['Saison', 'Equipe 1'])['Résultat'].rolling(window=6, min_periods=2).sum().reset_index(drop=True)-dataframe_final['Résultat']
dataframe_final['Forme 2'] = dataframe_final.merge(dataframe_final, how='left', left_on=['Saison', 'Journée', 'Equipe 1'], right_on=['Saison', 'Journée', 'Equipe 2'])['Forme 1_y']
dataframe_final['Historique'] = dataframe_final.groupby(['Equipe 1', 'Equipe 2'])['Résultat'].cumsum() - dataframe_final['Résultat']

dataframe_final.to_csv('dataframe_final.csv', encoding = 'utf-8', index=False)

model=LinearRegression()
dataframe_regression = dataframe_final.dropna().copy()
dataframe_regression = dataframe_regression[dataframe_regression['Journée'] > 0]
x1 = dataframe_regression[["Classement Equipe 1",  "Classement Equipe 2",  "Moyenne_BM par 1", "Moyenne_BM par 1 selon Lieu", "Moyenne_BE par 1", "Moyenne_BE par 1 selon Lieu","Moyenne_BM par 2", "Moyenne_BM par 2 selon Lieu", "Moyenne_BE par 2", "Moyenne_BE par 2 selon Lieu", 'Forme 1', 'Forme 2', 'Historique']]
y1 = dataframe_regression[["Buts Equipe 1"]]
model.fit(x1,y1)
y_pred1 =(model.predict(x1))
dataframe_regression ['pred_buts 1'] = y_pred1 
dataframe_regression['residuals 1'] = dataframe_regression['pred_buts 1'] - dataframe_regression['Buts Equipe 1']


x2 = dataframe_regression[["Classement Equipe 1",  "Classement Equipe 2",  "Moyenne_BM par 1", "Moyenne_BM par 1 selon Lieu", "Moyenne_BE par 1", "Moyenne_BE par 1 selon Lieu","Moyenne_BM par 2", "Moyenne_BM par 2 selon Lieu", "Moyenne_BE par 2", "Moyenne_BE par 2 selon Lieu", 'Forme 1', 'Forme 2', 'Historique']]
y2 = dataframe_regression[["Buts Equipe 2"]]
model.fit(x2,y2)
y_pred2 = (model.predict(x2))
dataframe_regression ['pred_buts 2'] = y_pred2
dataframe_regression['residuals 2'] = dataframe_regression['pred_buts 2'] - dataframe_regression['Buts Equipe 2']

conditions = [
    (dataframe_regression['pred_buts 1'] > dataframe_regression['pred_buts 2']),
    (dataframe_regression['pred_buts 1'] < dataframe_regression['pred_buts 2'])
]
valeurs = [1, -1]
dataframe_regression['Résultat prévu'] = 0
dataframe_regression['Résultat prévu'] = np.select(conditions, valeurs)
dataframe_regression['Bon_résultat'] = dataframe_regression['Résultat'] == dataframe_regression['Résultat prévu']
print(dataframe_regression.groupby('Résultat prévu')['Bon_résultat'].value_counts())
print(dataframe_regression['Bon_résultat'].value_counts())


model=LinearRegression()
x1 = dataframe_regression[["Classement Equipe 1",  "Classement Equipe 2",  "Moyenne_BM par 1", "Moyenne_BM par 1 selon Lieu", "Moyenne_BE par 1", "Moyenne_BE par 1 selon Lieu","Moyenne_BM par 2", "Moyenne_BM par 2 selon Lieu", "Moyenne_BE par 2", "Moyenne_BE par 2 selon Lieu", 'Forme 1', 'Forme 2', 'Historique']]
y1 = dataframe_regression[["Résultat"]]
model.fit(x1,y1)
y_pred1 =(model.predict(x1))
dataframe_regression ['pred'] = y_pred1

conditions = [
    (dataframe_regression['pred'] > 0.01),
    (dataframe_regression['pred'] < -0.01)
]
valeurs = [1, -1]
dataframe_regression['pred_résultat'] = 0
dataframe_regression['pred_résultat'] = np.select(conditions, valeurs)

dataframe_regression['Bon résultat'] = dataframe_regression['Résultat'] == dataframe_regression['pred_résultat']
print(dataframe_regression.groupby('pred_résultat')['Bon résultat'].value_counts())
print(dataframe_regression['Bon résultat'].value_counts())


dataframe_regression['Test'] = (dataframe_regression['Résultat'] == dataframe_regression['pred_résultat']) | (dataframe_regression['Résultat'] == dataframe_regression['Résultat prévu'])
print(dataframe_regression.groupby('Résultat')['Test'].value_counts())
print(dataframe_regression['Test'].value_counts())

dataframe_regression.to_excel('dataframe_regression.xlsx', index=False)


X_train = dataframe_regression[["Classement Equipe 1",  "Classement Equipe 2",  "Moyenne_BM par 1", "Moyenne_BM par 1 selon Lieu", "Moyenne_BE par 1", "Moyenne_BE par 1 selon Lieu","Moyenne_BM par 2", "Moyenne_BM par 2 selon Lieu", "Moyenne_BE par 2", "Moyenne_BE par 2 selon Lieu", 'Forme 1', 'Forme 2', 'Historique']][~(dataframe_regression['Saison'] == '2022-2023')]
Y_train = dataframe_regression["Résultat"][~(dataframe_regression['Saison'] == '2022-2023')]
Y_test = dataframe_regression[dataframe_regression['Saison'] == '2022-2023']['Résultat']
X_test = dataframe_regression[dataframe_regression['Saison'] == '2022-2023'][["Classement Equipe 1",  "Classement Equipe 2",  "Moyenne_BM par 1", "Moyenne_BM par 1 selon Lieu", "Moyenne_BE par 1", "Moyenne_BE par 1 selon Lieu","Moyenne_BM par 2", "Moyenne_BM par 2 selon Lieu", "Moyenne_BE par 2", "Moyenne_BE par 2 selon Lieu", 'Forme 1', 'Forme 2', 'Historique']]


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

svm_model = svm.SVC()

param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['linear', 'rbf', 'poly'],
              'gamma': ['scale', 'auto', 0.1, 1, 10]}

grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, Y_train)

print("Meilleurs hyperparamètres trouvés :")
print(grid_search.best_params_)

best_svm_model = grid_search.best_estimator_

Y_pred = best_svm_model.predict(X_test)

classification_report_result = classification_report(Y_test, Y_pred)

print("Classification Report sur l'ensemble de test:\n", classification_report_result)
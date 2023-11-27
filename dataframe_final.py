import pandas as pd
import numpy as np 
import openpyxl
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

dataframe_final['-Résultat'] = -1*dataframe_final['Résultat']
dataframe_final_copy = dataframe_final.copy()
dataframe_final = pd.merge(dataframe_final,dataframe_final_copy.groupby(['Saison','Domicile'])['Résultat'].rolling(window=5, min_periods=1).sum().reset_index().drop(columns = {'Saison','Domicile'}).rename(columns = {'Résultat':'Forme D'}), left_index=True, right_on='level_2', how='left').drop(columns = {'level_2'}).reset_index(drop=True)
dataframe_final = pd.merge(dataframe_final,dataframe_final_copy.groupby(['Saison','Extérieur'])['-Résultat'].rolling(window=5, min_periods=1).sum().reset_index().drop(columns = {'Saison','Extérieur'}).rename(columns = {'-Résultat':'Forme E'}), left_index=True, right_on='level_2', how='left').drop(columns = {'level_2'}).reset_index(drop=True)
dataframe_final['Forme D'] = dataframe_final['Forme D'] - dataframe_final['Résultat']
dataframe_final['Forme E'] = dataframe_final['Forme E'] - dataframe_final['-Résultat']
dataframe_final['Moyenne_BM par D à d'] = (dataframe_final.groupby(['Saison', 'Domicile'])['Buts domicile'].cumsum() - dataframe_final['Buts domicile']) / (dataframe_final.groupby(['Saison', 'Domicile'])['Journée'].cumcount())
dataframe_final['Moyenne_BE par D à d'] = (dataframe_final.groupby(['Saison', 'Domicile'])['Buts extérieur'].cumsum() - dataframe_final['Buts extérieur']) / (dataframe_final.groupby(['Saison', 'Domicile'])['Journée'].cumcount())
dataframe_final["Moyenne_BM par E à e"] = (dataframe_final.groupby(['Saison', 'Extérieur'])['Buts extérieur'].cumsum() - dataframe_final['Buts extérieur'])/ (dataframe_final.groupby(['Saison', 'Extérieur'])['Journée'].cumcount())
dataframe_final["Moyenne_BE par E à e"] = (dataframe_final.groupby(['Saison', 'Extérieur'])['Buts domicile'].cumsum() - dataframe_final['Buts domicile']) / (dataframe_final.groupby(['Saison', 'Extérieur'])['Journée'].cumcount())

model=LinearRegression()
dataframe_regression = dataframe_final.dropna().copy()
dataframe_regression = dataframe_regression[dataframe_regression['Journée'] > 30]
x1 = dataframe_regression[["Classement D",  "Classement E",  "Moyenne_BM par D à d", "Moyenne_BE par E à e", 'Forme D', 'Forme E']]
y1 = dataframe_regression[["Buts domicile"]]
model.fit(x1,y1)
y_pred1 =(model.predict(x1))
dataframe_regression ['pred_buts D'] = y_pred1 
dataframe_regression['residuals 1'] = dataframe_regression['pred_buts D'] - dataframe_regression['Buts domicile']

x2 = dataframe_regression[["Classement D",  "Classement E",  "Moyenne_BE par D à d", "Moyenne_BM par E à e",'Forme D', 'Forme E']]
y2 = dataframe_regression[["Buts extérieur"]]
model.fit(x2,y2)
y_pred2 = (model.predict(x2))
dataframe_regression ['pred_buts E'] = y_pred2
dataframe_regression['residuals 2'] = dataframe_regression['pred_buts E'] - dataframe_regression['Buts extérieur']

conditions = [
    (dataframe_regression['pred_buts D'] > dataframe_regression['pred_buts E'] + 0),
    (dataframe_regression['pred_buts D'] + 0 < dataframe_regression['pred_buts E']),
]
valeurs = [1, -1]
dataframe_regression['Résultat prévu'] = 0
dataframe_regression['Résultat prévu'] = np.select(conditions, valeurs)
dataframe_regression['Bon_résultat'] = dataframe_regression['Résultat'] == dataframe_regression['Résultat prévu']
print(dataframe_regression.groupby('Résultat prévu')['Bon_résultat'].value_counts())
print(dataframe_regression['Bon_résultat'].value_counts())

dataframe_regression.to_excel('dataframe_regression.xlsx', index=False)
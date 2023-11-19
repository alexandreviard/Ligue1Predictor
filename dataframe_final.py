import pandas as pd
import numpy as np 
pd.set_option('display.max_rows', 100)    
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
dataframe_final['Classement Domicile'] = np.select(conditions, valeurs)
dataframe_final['Classement Domicile'] = dataframe_final['Classement Domicile'].replace({0: np.nan}).astype(pd.Int64Dtype())


df_merge2 = pd.merge(dataframe_final, dataframe_classement, left_on=['Saison', 'Extérieur'], right_on=['Saison', 'Équipes'], how='left')
conditions = [df_merge2['Journée'] == (i + 1) for i in range(1, max(df_merge2['Journée']) + 1)]
valeurs = [df_merge2[f'J{i}'] for i in range(1, max(df_merge2['Journée']) + 1)]
dataframe_final['Classement Extérieur'] = np.select(conditions, valeurs)
dataframe_final['Classement Extérieur'] = dataframe_final['Classement Extérieur'].replace({0: np.nan}).astype(pd.Int64Dtype())

dataframe_final['Moyenne_BM par Domicile à domicile'] = (dataframe_final.groupby(['Saison', 'Domicile'])['Buts domicile'].cumsum() - dataframe_final['Buts domicile']) / (dataframe_final.groupby(['Saison', 'Domicile'])['Journée'].cumcount())
dataframe_final['Moyenne_BE par Domicile à domicile'] = (dataframe_final.groupby(['Saison', 'Domicile'])['Buts extérieur'].cumsum() - dataframe_final['Buts extérieur']) / (dataframe_final.groupby(['Saison', 'Domicile'])['Journée'].cumcount())
dataframe_final["Moyenne_BM par Extérieur à l'extérieur"] = (dataframe_final.groupby(['Saison', 'Extérieur'])['Buts extérieur'].cumsum() - dataframe_final['Buts extérieur'])/ (dataframe_final.groupby(['Saison', 'Extérieur'])['Journée'].cumcount())
dataframe_final["Moyenne_BE par Extérieur à l'extérieur"] = (dataframe_final.groupby(['Saison', 'Extérieur'])['Buts domicile'].cumsum() - dataframe_final['Buts domicile']) / (dataframe_final.groupby(['Saison', 'Extérieur'])['Journée'].cumcount())


print(dataframe_final.head(100))




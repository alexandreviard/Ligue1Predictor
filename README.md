# Projet Python - Scraping et Analyse de la Ligue 1

## Description
Ce projet, réalisé par Alexandre Viard et Vincent Graillat, étudiants en 2ème année à l'ENSAE, a pour objectif d'établir un modèle essayant de prédire les résultats des matchs de Ligue 1. Il se concentre sur le scraping des statistiques de la Ligue 1 depuis fbref. Il offre une riche base de données comprenant des dizaines de colonnes de statistiques pour chaque match et chaque équipe. L'objectif est une automatisation complète, avec un script Python quotidien qui met à jour la base avec les derniers résultats et génère des prévisions pour les prochains matchs.

## Installation
Clonez le dépôt GitHub et installez les dépendances via `requirements.txt`.

## Usage
Ouvrez `Rapport_projet_python.ipynb` pour accéder à notre rapport présentant deux approches mises en place. Ce fichier s'appuie sur les autres fichiers .py du dossier, dans lesquels se trouvent les fonctions utilisées dans le rapport. Se trouve aussi dans le dossier, les bases de données enregistrées au format csv pour éviter le scrapping trop long, notamment dans la deuxième approche. A la fin du rapport vous pourrez trouver une interface codée avec Tkinter qui vous permettra de manipuler la base.

## Automatisation
Le scraping est exécuté automatiquement chaque nuit à minuit pour une mise à jour des résultats et un téléchargement des futures rencontres.

## Analyse de Données
Le notebook principal détaille la construction et l'exploitation de la base de données, et offre un aperçu du développement de l'interface utilisateur.

## Contribution
Nous encourageons toute contribution visant à améliorer et à enrichir ce projet.

## Licence
Ce projet est ouvert et réalisé dans un esprit de collaboration et d'apprentissage académique. 

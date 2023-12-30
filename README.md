# Projet Python - Scraping et Analyse de la Ligue 1

## Description
Ce projet, réalisé par Alexandre Viard et Vincent Graillat, étudiants en 2ème année à l'ENSAE, vise à établir un modèle de prédiction des résultats des matchs de Ligue 1. Il se focalise sur le scraping des statistiques de la Ligue 1 depuis fbref, fournissant une base de données riche comprenant des dizaines de colonnes de statistiques pour chaque match et chaque équipe. L'objectif est d'automatiser entièrement le processus, avec un script Python quotidien qui met à jour la base avec les derniers résultats et génère des prévisions pour les prochains matchs.

## Installation
Clonez le dépôt GitHub et installez les dépendances via **`requirements.txt`**.

## Usage
Ouvrez **`Rapport_projet_python.ipynb`** pour consulter notre rapport présentant deux approches. Ce fichier s'appuie sur d'autres fichiers **.py** du dossier contenant les fonctions utilisées dans le rapport. Le dossier inclut également des bases de données enregistrées au format csv pour éviter un scraping trop long, en particulier pour la deuxième approche. À la fin du rapport, vous trouverez une interface développée avec Tkinter permettant de manipuler la base.

## Automatisation
Le scraping est exécuté automatiquement chaque nuit à minuit pour une mise à jour des résultats et un téléchargement des prochaines rencontres. Ceci est réalisé via le fichier **`script_github.py`**, lancé chaque soir à minuit par un workflow GitHub. Une automatisation du scraping des prochaines journées était essentielle dans un contexte d'évaluation sportive.

## Prévisions Dynamiques Automatiques
En plus de l'automatisation des bases de données, le script est conçu pour prédire automatiquement les futures journées de Ligue 1 sur plusieurs indicateurs : le résultat du match ("W_Home", "W_Away", "D") et un indicateur courant chez les bookmakers, "Moins de 2.5 Buts", avec la probabilité de réalisation de l'événement prédit. Cela est enregistré automatiquement dans le fichier **`results.csv`** et les prédictions sont basées sur un modèle de forêt aléatoire, qui a offert les meilleures performances.

## Analyse de Données
Le notebook principal détaille la construction et l'utilisation des deux bases de données avec des statistiques détaillées, la méthodologie de construction des modèles et des comparaisons de performance. Il offre également un aperçu du développement de l'interface utilisateur Tkinter, qui permet de consulter des statistiques sur les équipes, d'accéder aux prévisions faites automatiquement et de visualiser des données historiques.

## Contribution
Nous encourageons toute contribution visant à améliorer et à enrichir ce projet. Ce projet est ouvert et réalisé dans un esprit de collaboration et d'apprentissage académique.

# DSLR - Data Science x Logistic Regression

Projet d'analyse de données et d'apprentissage automatique inspiré de l'univers Harry Potter.

## Description

Ce projet consiste à analyser un dataset contenant des informations sur les étudiants de Poudlard et à prédire leur maison d'appartenance en utilisant la régression logistique.

## Dataset

Le projet contient deux fichiers de données :
- `datasets/dataset_train.csv` : Dataset d'entraînement (1600 étudiants)
- `datasets/dataset_test.csv` : Dataset de test (400 étudiants)

### Structure des données

Le dataset contient les colonnes suivantes :
- **Index** : Identifiant unique de l'étudiant (non analysé)
- **Hogwarts House** : Maison d'appartenance (Gryffindor, Slytherin, Ravenclaw, Hufflepuff)
- **First Name** : Prénom de l'étudiant (non analysé)
- **Last Name** : Nom de famille de l'étudiant (non analysé)
- **Birthday** : Date de naissance (non analysé)
- **Best Hand** : Main dominante - Left/Right (non analysé)
- **Arithmancy** : Note en Arithmancie
- **Astronomy** : Note en Astronomie
- **Herbology** : Note en Botanique
- **Defense Against the Dark Arts** : Note en Défense contre les Forces du Mal
- **Divination** : Note en Divination
- **Muggle Studies** : Note en Études des Moldus
- **Ancient Runes** : Note en Runes anciennes
- **History of Magic** : Note en Histoire de la Magie
- **Transfiguration** : Note en Métamorphose
- **Potions** : Note en Potions
- **Care of Magical Creatures** : Note en Soins aux créatures magiques
- **Charms** : Note en Sortilèges
- **Flying** : Note en Vol

### Observations sur les données

- **Taille** : 1600 étudiants dans le dataset d'entraînement
- **Features numériques** : 13 matières avec des notes numériques
- **Valeurs manquantes** : Présentes dans plusieurs colonnes (visible via le Count)
- **Plages de valeurs variées** : 
  - Arithmancy : large plage (de -24370 à 104956)
  - Astronomy : -966 à 1016
  - La plupart des autres matières : échelles plus petites (-10 à +15 environ)
- **Types de données** : 
  - Numériques : Index, toutes les notes de matières
  - Catégorielles : Hogwarts House, Best Hand
  - Textuelles : First Name, Last Name
  - Dates : Birthday

## Programme describe.py

### Usage

```bash
python describe.py <dataset.csv>
```

### Description

Ce programme affiche des statistiques descriptives pour toutes les features numériques du dataset, similaire à la fonction `pandas.DataFrame.describe()`.

Les statistiques calculées sont :
- **Count** : Nombre de valeurs non-nulles
- **Mean** : Moyenne arithmétique
- **Std** : Écart-type (standard deviation)
- **Min** : Valeur minimale
- **25%** : Premier quartile (25ème percentile)
- **50%** : Médiane (50ème percentile)
- **75%** : Troisième quartile (75ème percentile)
- **Max** : Valeur maximale

### Exemple

```bash
python describe.py datasets/dataset_train.csv
```

Sortie :
```
            Arithmancy    Astronomy   Herbology     Defense  Divination        Muggle       Runes    History      Transf.    Potions  Creatures       Charms       Flying
Count             1566         1568        1567        1569        1561          1565        1565       1557         1566       1570       1560         1600         1600
Mean      49634.570243    39.797131    1.141020   -0.387863    3.153910   -224.589915  495.747970   2.963095  1030.096946   5.950373  -0.053427  -243.374409    21.958012
Std       16674.479577   520.132330    5.218016    5.211132    4.153970    486.189433  106.251202   4.424353    44.111025   3.146852   0.971146     8.780895    97.601087
...
```

### Implémentation

Le programme est écrit en Python pur sans utiliser pandas ou numpy, avec des implémentations personnalisées pour :
- Calcul de la moyenne
- Calcul de l'écart-type
- Calcul des percentiles (méthode d'interpolation linéaire)
- Gestion des valeurs manquantes
- Exclusion automatique des colonnes non pertinentes (Index, Name, etc.)
- Raccourcissement des noms de colonnes pour un affichage compact
- Largeur de colonnes adaptative selon le contenu

## Auteur

Kenny Duong

## Date

17 décembre 2025

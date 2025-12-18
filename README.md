# DSLR - Data Science x Logistic Regression

Projet d'analyse de données et d'apprentissage automatique inspiré de l'univers Harry Potter.

## Description

Ce projet consiste à analyser un dataset contenant des informations sur les étudiants de Poudlard et à prédire leur maison d'appartenance en utilisant la régression logistique.

## Installation

### Prérequis
- Python 3.9 ou supérieur
- pip

### Configuration de l'environnement virtuel

```bash
# Cloner le repository
git clone https://github.com/kennyydng/dslr.git
cd dslr

# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
# Sur macOS/Linux :
source venv/bin/activate
# Sur Windows :
# venv\Scripts\activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### Dépendances
- matplotlib >= 3.5.0 : Visualisation de données
- numpy >= 1.21.0 : Calculs numériques
- seaborn >= 0.12.0 : Visualisations statistiques avancées
- pandas >= 1.3.0 : Manipulation de données (pour les visualisations uniquement)

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
# Avec l'environnement virtuel activé
python src/describe.py <dataset.csv>

# Exemple
python src/describe.py datasets/dataset_train.csv
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
python src/describe.py datasets/dataset_train.csv
```

Sortie :
```
            Arithmancy    Astronomy   Herbology  Defense Against  Divination        Muggle     Ancient  History of  Transfiguration    Potions  Care of Magical       Charms       Flying
                                                   the Dark Arts                   Studies       Runes       Magic                                    Creatures                          
Count             1566         1568        1567             1569        1561          1565        1565        1557             1566       1570             1560         1600         1600
Mean      49634.570243    39.797131    1.141020        -0.387863    3.153910   -224.589915  495.747970    2.963095      1030.096946   5.950373        -0.053427  -243.374409    21.958012
Std       16674.479577   520.132330    5.218016         5.211132    4.153970    486.189433  106.251202    4.424353        44.111025   3.146852         0.971146     8.780895    97.601087
...
```

## Scripts de visualisation

**Note**: Si `python` ne fonctionne pas, utilisez `./venv/bin/python` ou `python3` à la place.

### histogram.py - Analyse d'homogénéité

Affiche des histogrammes pour identifier le cours avec la distribution de scores la plus homogène entre les quatre maisons.

```bash
# Méthode 1 (avec venv activé)
source venv/bin/activate
python src/histogram.py datasets/dataset_train.csv

# Méthode 2 (chemin direct)
./venv/bin/python src/histogram.py datasets/dataset_train.csv
```

**Question** : Quel cours a une distribution de scores homogène entre toutes les maisons ?  
**Réponse** : Care of Magical Creatures

### scatter_plot.py - Analyse de similarité

Affiche des scatter plots pour trouver les deux features les plus similaires (fortement corrélées).

```bash
./venv/bin/python src/scatter_plot.py datasets/dataset_train.csv
```

**Question** : Quelles sont les deux features les plus similaires ?  
**Réponse** : Astronomy et Defense Against the Dark Arts (corrélation parfaite de 1.0)

### pair_plot.py - Sélection de features

Affiche un pair plot et une matrice de corrélation pour aider à sélectionner les meilleures features pour la régression logistique.

```bash
./venv/bin/python src/pair_plot.py datasets/dataset_train.csv
```

**Question** : Quelles features utiliser pour la régression logistique ?  
**Réponse** : Les 5 features les plus discriminantes sont :
1. Defense Against the Dark Arts
2. Astronomy
3. Divination
4. Charms
5. Ancient Runes

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

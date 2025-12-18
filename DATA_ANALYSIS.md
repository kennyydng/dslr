# Analyse des données - DSLR

## Vue d'ensemble du dataset

### Dataset d'entraînement (dataset_train.csv)

**Taille** : 1600 étudiants
**Features numériques analysées** : 13 matières (Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying)

### Caractéristiques des données

#### 1. Variables catégorielles
- **Hogwarts House** : 4 maisons (Gryffindor, Slytherin, Ravenclaw, Hufflepuff)
- **Best Hand** : Main dominante (Left, Right)

#### 2. Variables textuelles
- **First Name** : Prénom (1600 valeurs uniques)
- **Last Name** : Nom de famille

#### 3. Variables temporelles
- **Birthday** : Date de naissance

#### 4. Variables numériques (13 matières)

**Note** : Le programme `describe.py` exclut automatiquement les colonnes non pertinentes (Index, First Name, Last Name, Birthday, Best Hand, Hogwarts House) pour ne garder que les features numériques des matières.

| Feature | Count | Mean | Std | Min | Max | Valeurs manquantes |
|---------|-------|------|-----|-----|-----|--------------------|
| Arithmancy | 1566 | 49634.6 | 16674.5 | -24370 | 104956 | 34 |
| Astronomy | 1568 | 39.8 | 520.1 | -966.7 | 1016.2 | 32 |
| Herbology | 1567 | 1.14 | 5.22 | -10.30 | 11.61 | 33 |
| Defense Against the Dark Arts | 1569 | -0.39 | 5.21 | -10.16 | 9.67 | 31 |
| Divination | 1561 | 3.15 | 4.15 | -8.73 | 10.03 | 39 |
| Muggle Studies | 1565 | -224.59 | 486.19 | -1086.50 | 1092.39 | 35 |
| Ancient Runes | 1565 | 495.75 | 106.25 | 283.87 | 745.40 | 35 |
| History of Magic | 1557 | 2.96 | 4.42 | -8.86 | 11.89 | 43 |
| Transfiguration | 1566 | 1030.10 | 44.11 | 906.63 | 1098.96 | 34 |
| Potions | 1570 | 5.95 | 3.15 | -4.70 | 13.54 | 30 |
| Care of Magical Creatures | 1560 | -0.05 | 0.97 | -3.31 | 3.06 | 40 |
| Charms | 1600 | -243.37 | 8.78 | -261.05 | -225.43 | 0 |
| Flying | 1600 | 21.96 | 97.60 | -181.47 | 279.07 | 0 |

## Observations importantes

### 1. Valeurs manquantes
- Toutes les matières ont des valeurs manquantes (entre 30 et 43 valeurs manquantes)
- **Charms** et **Flying** sont les seules matières sans valeurs manquantes
- **History of Magic** a le plus de valeurs manquantes (43)

### 2. Échelles de valeurs très différentes
Les matières ont des échelles très différentes :

**Très grande échelle** :
- **Arithmancy** : [-24370, 104956] - amplitude ~129000
- **Astronomy** : [-966.7, 1016.2] - amplitude ~1983

**Grande échelle** :
- **Transfiguration** : [906.63, 1098.96] - amplitude ~192
- **Ancient Runes** : [283.87, 745.40] - amplitude ~461
- **Muggle Studies** : [-1086.50, 1092.39] - amplitude ~2179

**Échelle moyenne** :
- **Flying** : [-181.47, 279.07] - amplitude ~460

**Petite échelle** :
- Toutes les autres matières : amplitude entre 10 et 25

### 3. Distribution des données

**Matières avec moyennes proches de 0** :
- Care of Magical Creatures : -0.05 ± 0.97
- Defense Against the Dark Arts : -0.39 ± 5.21
- Herbology : 1.14 ± 5.22
- History of Magic : 2.96 ± 4.42

**Matières avec moyennes négatives importantes** :
- Charms : -243.37 ± 8.78
- Muggle Studies : -224.59 ± 486.19

**Matières avec moyennes positives importantes** :
- Transfiguration : 1030.10 ± 44.11
- Ancient Runes : 495.75 ± 106.25
- Arithmancy : 49634.57 ± 16674.48

### 4. Variabilité (Coefficient de variation)

**Faible variabilité** (données homogènes) :
- Charms : CV = 3.6%
- Transfiguration : CV = 4.3%
- Potions : CV = 52.9%

**Forte variabilité** (données hétérogènes) :
- Arithmancy : CV = 33.6%
- Astronomy : CV = 1306% (très dispersé autour de la moyenne proche de 0)
- Muggle Studies : CV = -216%

## Recommandations pour le preprocessing

1. **Normalisation/Standardisation** : Indispensable vu les échelles très différentes
   - Z-score standardization pour la plupart des features
   - Min-Max scaling possible pour certaines features

2. **Gestion des valeurs manquantes** :
   - Imputation par la moyenne/médiane par maison
   - Ou suppression des lignes (perte de ~3% des données max)

3. **Analyse de corrélation** :
   - Étudier les corrélations entre matières
   - Identifier les features les plus discriminantes pour chaque maison

4. **Détection d'outliers** :
   - Particulièrement pour Arithmancy (très large plage)
   - Astronomy (forte variance)

5. **Feature engineering** :
   - Possibilité de créer des features combinées
   - Ratio entre certaines matières
   - Catégorisation de Best Hand et Birthday

## Prochaines étapes

1. Visualisation des distributions par maison
2. Analyse de corrélation
3. Sélection des features pertinentes
4. Implémentation de la régression logistique

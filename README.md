# DSLR - Data Science x Logistic Regression

Analyse de données et classification multi-classe avec régression logistique.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### 1. Analyse descriptive
```bash
python src/describe.py datasets/dataset_train.csv
```
Affiche les statistiques (Count, Mean, Std, Min, 25%, 50%, 75%, Max, Range, IQR, Skewness, Kurtosis).

### 2. Visualisations

**Histogrammes** (cours le plus homogène entre maisons):
```bash
python src/histogram.py datasets/dataset_train.csv
```
→ Réponse : Care of Magical Creatures

**Scatter plots** (features les plus corrélées):
```bash
python src/scatter_plot.py datasets/dataset_train.csv
```
→ Réponse : Astronomy & Defense Against the Dark Arts (r=1.0)

**Pair plot** (sélection de features):
```bash
python src/pair_plot.py datasets/dataset_train.csv
```
→ Top 5 features pour la classification

### 3. Entraînement

```bash
python src/logreg_train.py datasets/dataset_train.csv
```
- Entraîne 4 modèles (one-vs-all)
- Sauvegarde dans `weights.json`
- Précision : >98%

### 4. Prédiction

```bash
python src/logreg_predict.py datasets/dataset_test.csv weights.json
```
- Génère `houses.csv` avec les prédictions
- Affiche automatiquement les graphiques de répartition
- Affiche les statistiques par maison

## Bonus

### Algorithmes d'optimisation alternatifs

**Stochastic Gradient Descent** (1 exemple à la fois):
```bash
python bonus/logreg_train_sgd.py datasets/dataset_train.csv
```
→ Génère `weights_sgd.json`

**Mini-Batch Gradient Descent** (batches de 32-64):
```bash
python bonus/logreg_train_minibatch.py datasets/dataset_train.csv 64
```
→ Génère `weights_minibatch.json`

### Comparaison des méthodes

```bash
python bonus/compare_methods.py
```
Affiche la comparaison des poids et performances des 3 algorithmes.

### Démonstration complète

```bash
./bonus/run_all_bonus.sh
```
Lance tous les bonus de façon interactive.

## Structure

```
dslr/
├── src/
│   ├── describe.py          # Statistiques
│   ├── histogram.py         # Homogénéité
│   ├── scatter_plot.py      # Corrélation
│   ├── pair_plot.py         # Sélection features
│   ├── logreg_train.py      # Entraînement
│   └── logreg_predict.py    # Prédiction + Visualisation
├── bonus/
│   ├── logreg_train_sgd.py  # SGD
│   ├── logreg_train_minibatch.py  # Mini-Batch
│   ├── compare_methods.py   # Comparaison
│   └── run_all_bonus.sh     # Démo
└── datasets/
    ├── dataset_train.csv
    └── dataset_test.csv
```

## Résultats

| Script | Sortie | Info |
|--------|--------|------|
| `describe.py` | 12 stats × 13 features | Count, Mean, Std, Min, 25%, 50%, 75%, Max, Range, IQR, Skewness, Kurtosis |
| `histogram.py` | Graphiques | Care of Magical Creatures = plus homogène |
| `scatter_plot.py` | Graphiques | Astronomy & Defense = corrélation parfaite (r=1.0) |
| `pair_plot.py` | Graphiques + scores | Top 5 features identifiées |
| `logreg_train.py` | `weights.json` | >98% précision (4 modèles one-vs-all) |
| `logreg_predict.py` | `houses.csv` + graphiques | 362/400 prédictions + stats détaillées |
| **Bonus SGD** | `weights_sgd.json` | >98% précision (147k mises à jour) |
| **Bonus Mini-Batch** | `weights_minibatch.json` | >98% précision (2.3k mises à jour) |
| **Bonus Comparaison** | Tableau comparatif | Poids et performances des 3 algorithmes |

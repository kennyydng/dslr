# Bonus DSLR

## 1. Statistiques avancées (describe.py)

### Nouvelles statistiques
- **Range** : `Max - Min` (étendue totale)
- **Skewness** : Asymétrie de la distribution
  - < 0 : asymétrique gauche
  - ≈ 0 : symétrique
  - \> 0 : asymétrique droite
- **Kurtosis** : Épaisseur des queues
  - < 0 : queues légères
  - ≈ 0 : normale
  - \> 0 : queues lourdes

### Usage
```bash
python src/describe.py datasets/dataset_train.csv
```
→ Affiche 11 statistiques au lieu de 8

---

## 2. Stochastic Gradient Descent (SGD)

### Principe
Met à jour les poids après **chaque exemple** (au lieu de tout le batch).

### Caractéristiques
- Learning rate : 0.01
- Époques : 100
- Mises à jour : 1470 par époque = **147,000 total**
- Shuffle aléatoire à chaque époque

### Avantages / Inconvénients
✅ Convergence rapide  
✅ Peut échapper aux minima locaux  
✅ Faible mémoire  
❌ Convergence bruitée  

### Usage
```bash
python bonus/logreg_train_sgd.py datasets/dataset_train.csv
```
→ Génère `weights_sgd.json` (>98% précision)

---

## 3. Mini-Batch Gradient Descent

### Principe
Met à jour les poids par **mini-batches** (groupes de 32-64 exemples).

### Caractéristiques
- Learning rate : 0.1
- Époques : 100
- Batch size : 32-64 (configurable)
- Mises à jour : ~23 par époque = **2,300 total**

### Avantages / Inconvénients
✅ Meilleur compromis vitesse/stabilité  
✅ Parallélisable  
✅ Convergence stable  
✅ **Recommandé pour la production**  

### Usage
```bash
python bonus/logreg_train_minibatch.py datasets/dataset_train.csv 64
```
→ Génère `weights_minibatch.json` (>98% précision)

---

## 4. Comparaison des algorithmes

### Script de comparaison
```bash
python bonus/compare_methods.py
```

### Tableau comparatif

| Critère | Batch GD | SGD | Mini-Batch |
|---------|----------|-----|------------|
| **Mises à jour/époque** | 1 | 1470 | ~23 |
| **Learning rate** | 0.5 | 0.01 | 0.1 |
| **Total mises à jour** | 1,000 | 147,000 | 2,300 |
| **Convergence** | Lente stable | Rapide bruitée | Équilibrée |
| **Mémoire** | Dataset complet | 1 exemple | 1 batch |
| **Précision** | >98% | >98% | >98% |
| **Fichier** | `weights.json` | `weights_sgd.json` | `weights_minibatch.json` |

### Résultat
Les 3 méthodes atteignent **>98% de précision** mais avec des caractéristiques différentes.

---

## Démonstration complète

### Script interactif
```bash
./bonus/run_all_bonus.sh
```

Ce script :
1. Affiche les statistiques avancées
2. Entraîne les 3 algorithmes
3. Compare les poids et performances
4. Génère les prédictions avec chaque méthode
5. Affiche un résumé complet

---

## Utilisation avec les bonus

### Workflow avec SGD
```bash
python bonus/logreg_train_sgd.py datasets/dataset_train.csv
python src/logreg_predict.py datasets/dataset_test.csv weights_sgd.json
```

### Workflow avec Mini-Batch
```bash
python bonus/logreg_train_minibatch.py datasets/dataset_train.csv 64
python src/logreg_predict.py datasets/dataset_test.csv weights_minibatch.json
```

### Comparaison finale
```bash
python bonus/compare_methods.py
```
→ Affiche les poids appris par chaque algorithme et leurs différences

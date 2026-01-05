#!/usr/bin/env python3
"""
logreg_train - Entraînement de modèles de régression logistique one-vs-all
pour prédire la maison de Poudlard des étudiants
"""

import sys
import csv
import json
import math

from utils import clamp, ft_length, ft_sum, mean, std, is_nan


def parse_float(value):
    """Convertit une valeur en float, retourne None si impossible"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def read_csv(filepath):
    """Lit un fichier CSV et retourne les données"""
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filepath}' n'existe pas.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}", file=sys.stderr)
        sys.exit(1)


def extract_features_and_labels(data, selected_features):
    """Extrait les features et labels du dataset"""
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    
    X = []
    y = []
    
    for row in data:
        house = row.get('Hogwarts House')
        if house not in houses:
            continue
        
        # Extraire les features
        features = []
        valid = True
        for feature in selected_features:
            value = parse_float(row.get(feature))
            if value is None:
                valid = False
                break
            features.append(value)
        
        if valid:
            X.append(features)
            y.append(house)
    
    return X, y


def normalize_features(X):
    """Normalise les features (standardisation)"""
    n_features = ft_length(X[0])
    means = []
    stds = []
    
    # Calculer moyenne et écart-type pour chaque feature
    for j in range(n_features):
        values = []
        for row in X:
            values.append(row[j])

        m = mean(values)
        s = std(values)
        if is_nan(s) or s == 0.0:
            s = 1.0

        means.append(m)
        stds.append(s)
    
    # Normaliser
    X_normalized = []
    for row in X:
        normalized_row = []
        j = 0
        while j < n_features:
            normalized_row.append((row[j] - means[j]) / stds[j])
            j += 1
        X_normalized.append(normalized_row)
    
    return X_normalized, means, stds


def sigmoid(z):
    """Fonction sigmoïde"""
    # Clipper pour éviter l'overflow
    z = clamp(z, -500, 500)
    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(X, weights):
    """Calcule la probabilité avec la régression logistique"""
    # z = w0 + w1*x1 + w2*x2 + ... (weights[0] est le biais)
    z = weights[0]  # Biais
    i = 0
    n = ft_length(X)
    while i < n:
        z += weights[i + 1] * X[i]
        i += 1
    return sigmoid(z)


def compute_cost(X, y_binary, weights):
    """Calcule le coût (log loss)"""
    m = ft_length(X)
    total_cost = 0.0
    
    for i in range(m):
        h = predict_probability(X[i], weights)
        # Éviter log(0)
        h = clamp(h, 1e-15, 1 - 1e-15)
        cost = -y_binary[i] * math.log(h) - (1 - y_binary[i]) * math.log(1 - h)
        total_cost += cost
    
    return total_cost / m


def gradient_descent(X, y_binary, learning_rate=0.1, iterations=1000):
    """Entraîne un modèle de régression logistique avec gradient descent"""
    m = ft_length(X)
    n_features = ft_length(X[0])
    
    # Initialiser les poids (biais + features)
    weights = [0.0] * (n_features + 1)
    
    for iteration in range(iterations):
        # Calculer les gradients
        gradients = [0.0] * (n_features + 1)
        
        for i in range(m):
            h = predict_probability(X[i], weights)
            error = h - y_binary[i]
            
            # Gradient pour le biais
            gradients[0] += error
            
            # Gradients pour les features
            for j in range(n_features):
                gradients[j + 1] += error * X[i][j]
        
        # Mettre à jour les poids
        j = 0
        w_n = ft_length(weights)
        while j < w_n:
            weights[j] -= (learning_rate / m) * gradients[j]
            j += 1
        
        # Afficher le coût périodiquement
        if (iteration + 1) % 100 == 0:
            cost = compute_cost(X, y_binary, weights)
            print(f"  Itération {iteration + 1}/{iterations}, Coût: {cost:.6f}")
    
    return weights


def train_one_vs_all(X, y, houses, learning_rate=0.1, iterations=1000):
    """Entraîne un classifieur one-vs-all pour chaque maison"""
    models = {}
    
    for house in houses:
        print(f"\nEntraînement du modèle pour {house}...")
        
        # Créer les labels binaires (1 si house, 0 sinon)
        y_binary = []
        for label in y:
            y_binary.append(1.0 if label == house else 0.0)
        
        # Entraîner le modèle
        weights = gradient_descent(X, y_binary, learning_rate, iterations)
        
        models[house] = weights
        
        # Calculer la précision sur le training set
        correct = 0
        i = 0
        m = ft_length(X)
        while i < m:
            prob = predict_probability(X[i], weights)
            if (prob >= 0.5 and y_binary[i] == 1) or (prob < 0.5 and y_binary[i] == 0):
                correct += 1
            i += 1
        accuracy = (correct / m * 100) if m > 0 else 0
        print(f"  Précision sur training set: {accuracy:.2f}%")
    
    return models


def save_weights(models, means, stds, selected_features, filepath='weights.json'):
    """Sauvegarde les poids et paramètres de normalisation"""
    data = {
        'features': selected_features,
        'normalization': {
            'means': means,
            'stds': stds
        },
        'models': {house: weights for house, weights in models.items()}
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Poids sauvegardés dans '{filepath}'")


def main():
    """Fonction principale"""
    if ft_length(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_train.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Features sélectionnées (basées sur l'analyse du pair_plot)
    selected_features = [
        'Astronomy',
        'Herbology',
        'Defense Against the Dark Arts',
        'Ancient Runes',
        'Charms'
    ]
    
    print("="*70)
    print("ENTRAÎNEMENT DE LA RÉGRESSION LOGISTIQUE ONE-VS-ALL")
    print("="*70)
    print(f"\nFeatures utilisées: {', '.join(selected_features)}")
    
    # Lire les données
    print(f"\nChargement du dataset: {filepath}")
    data = read_csv(filepath)
    print(f"  → {ft_length(data)} lignes chargées")
    
    # Extraire features et labels
    print("\nExtraction des features et labels...")
    X, y = extract_features_and_labels(data, selected_features)
    print(f"  → {ft_length(X)} exemples valides après nettoyage")
    
    # Normaliser les features
    print("\nNormalisation des features...")
    X_normalized, means, stds = normalize_features(X)
    print("  → Normalisation terminée")
    
    # Entraîner les modèles
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    learning_rate = 0.5
    iterations = 1000
    
    print(f"\nParamètres d'entraînement:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Itérations: {iterations}")
    
    models = train_one_vs_all(X_normalized, y, houses, learning_rate, iterations)
    
    # Sauvegarder les poids
    save_weights(models, means, stds, selected_features)
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*70)


if __name__ == "__main__":
    main()

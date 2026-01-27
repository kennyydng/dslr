#!/usr/bin/env python3
"""
logreg_train_sgd - Entraînement de modèles de régression logistique one-vs-all
avec Stochastic Gradient Descent (SGD) pour prédire la maison de Poudlard
"""

import sys
import json
import random

sys.path.insert(0, '../src')
from math_utils import (
    ft_sqrt, ft_exp, ft_log, parse_float, read_csv
)
from ml_utils import (
    normalize_features, sigmoid, predict_probability, 
    compute_cost, extract_features_and_labels, save_weights
)


def stochastic_gradient_descent(X, y_binary, learning_rate=0.1, epochs=100, verbose=True):
    """
    Entraîne un modèle de régression logistique avec Stochastic Gradient Descent (SGD)
    
    SGD met à jour les poids après chaque exemple individuel au lieu d'utiliser
    tout le batch. Cela permet une convergence plus rapide et peut aider à échapper
    aux minima locaux.
    
    Args:
        X: Données d'entraînement
        y_binary: Labels binaires
        learning_rate: Taux d'apprentissage
        epochs: Nombre d'époques (passages complets sur le dataset)
        verbose: Afficher les informations de progression
    """
    m = len(X)
    n_features = len(X[0])
    
    # Initialiser les poids (biais + features)
    weights = [0.0] * (n_features + 1)
    
    # Créer les indices pour le mélange
    indices = list(range(m))
    
    for epoch in range(epochs):
        # Mélanger les données à chaque époque
        random.shuffle(indices)
        
        # Mettre à jour les poids pour chaque exemple
        for idx in indices:
            i = indices[idx]
            
            # Calculer la prédiction pour cet exemple
            h = predict_probability(X[i], weights)
            error = h - y_binary[i]
            
            # Mettre à jour le biais
            weights[0] -= learning_rate * error
            
            # Mettre à jour les poids des features
            for j in range(n_features):
                weights[j + 1] -= learning_rate * error * X[i][j]
        
        # Afficher le coût périodiquement
        if verbose and (epoch + 1) % 10 == 0:
            cost = compute_cost(X, y_binary, weights)
            print(f"  Époque {epoch + 1}/{epochs}, Coût: {cost:.6f}")
    
    return weights


def train_one_vs_all(X, y, houses, learning_rate=0.1, epochs=100):
    """Entraîne un classifieur one-vs-all pour chaque maison avec SGD"""
    models = {}
    
    for house in houses:
        print(f"\nEntraînement du modèle pour {house} (SGD)...")
        
        # Créer les labels binaires (1 si house, 0 sinon)
        y_binary = [1.0 if label == house else 0.0 for label in y]
        
        # Entraîner le modèle avec SGD
        weights = stochastic_gradient_descent(X, y_binary, learning_rate, epochs)
        
        models[house] = weights
        
        # Calculer la précision sur le training set
        correct = 0
        for i in range(len(X)):
            prob = predict_probability(X[i], weights)
            if (prob >= 0.5 and y_binary[i] == 1) or (prob < 0.5 and y_binary[i] == 0):
                correct += 1
        accuracy = correct / len(X) * 100
        print(f"  Précision sur training set: {accuracy:.2f}%")
    
    return models


def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python logreg_train_sgd.py <dataset_train.csv>", file=sys.stderr)
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
    print("ENTRAÎNEMENT AVEC STOCHASTIC GRADIENT DESCENT (SGD)")
    print("="*70)
    print(f"\nFeatures utilisées: {', '.join(selected_features)}")
    
    # Lire les données
    print(f"\nChargement du dataset: {filepath}")
    data = read_csv(filepath)
    print(f"  → {len(data)} lignes chargées")
    
    # Extraire features et labels
    print("\nExtraction des features et labels...")
    X, y = extract_features_and_labels(data, selected_features)
    print(f"  → {len(X)} exemples valides après nettoyage")
    
    # Normaliser les features
    print("\nNormalisation des features...")
    X_normalized, means, stds = normalize_features(X)
    print("  → Normalisation terminée")
    
    # Entraîner les modèles
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    learning_rate = 0.01  # Plus petit learning rate pour SGD
    epochs = 100  # Nombre d'époques
    
    print(f"\nParamètres d'entraînement:")
    print(f"  - Méthode: Stochastic Gradient Descent (SGD)")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Époques: {epochs}")
    print(f"  - Mises à jour par époque: {len(X)} (une par exemple)")
    
    # Fixer la seed pour la reproductibilité
    random.seed(42)
    
    models = train_one_vs_all(X_normalized, y, houses, learning_rate, epochs)
    
    # Sauvegarder les poids
    save_weights(models, means, stds, selected_features, 'weights_sgd.json')
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*70)


if __name__ == "__main__":
    main()

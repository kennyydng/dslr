#!/usr/bin/env python3
"""
logreg_train_minibatch - Entraînement de modèles de régression logistique one-vs-all
avec Mini-Batch Gradient Descent pour prédire la maison de Poudlard
"""

import sys
import csv
import json
import random

sys.path.insert(0, '../src')
from math_utils import (
    ft_sqrt, ft_exp, ft_log, ft_ceil, parse_float, read_csv
)
from ml_utils import (
    normalize_features, sigmoid, predict_probability, 
    compute_cost, extract_features_and_labels, save_weights
)


def create_mini_batches(X, y_binary, batch_size):
    """Crée des mini-batches pour l'entraînement"""
    m = len(X)
    indices = list(range(m))
    random.shuffle(indices)
    
    mini_batches = []
    num_batches = ft_ceil(m / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, m)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = [X[idx] for idx in batch_indices]
        y_batch = [y_binary[idx] for idx in batch_indices]
        
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches


def minibatch_gradient_descent(X, y_binary, learning_rate=0.1, epochs=100, batch_size=32, verbose=True):
    """
    Entraîne un modèle de régression logistique avec Mini-Batch Gradient Descent
    
    Mini-Batch GD est un compromis entre Batch GD et SGD. Il met à jour les poids
    en utilisant des sous-ensembles du dataset, ce qui offre:
    - Une convergence plus stable que SGD
    - Une vitesse d'entraînement plus rapide que Batch GD
    - Une meilleure utilisation de la mémoire et du cache
    
    Args:
        X: Données d'entraînement
        y_binary: Labels binaires
        learning_rate: Taux d'apprentissage
        epochs: Nombre d'époques
        batch_size: Taille des mini-batches
        verbose: Afficher les informations de progression
    """
    m = len(X)
    n_features = len(X[0])
    
    # Initialiser les poids (biais + features)
    weights = [0.0] * (n_features + 1)
    
    for epoch in range(epochs):
        # Créer les mini-batches avec mélange
        mini_batches = create_mini_batches(X, y_binary, batch_size)
        
        # Entraîner sur chaque mini-batch
        for X_batch, y_batch in mini_batches:
            batch_m = len(X_batch)
            
            # Calculer les gradients pour ce mini-batch
            gradients = [0.0] * (n_features + 1)
            
            for i in range(batch_m):
                h = predict_probability(X_batch[i], weights)
                error = h - y_batch[i]
                
                # Gradient pour le biais
                gradients[0] += error
                
                # Gradients pour les features
                for j in range(n_features):
                    gradients[j + 1] += error * X_batch[i][j]
            
            # Mettre à jour les poids avec la moyenne des gradients du mini-batch
            for j in range(len(weights)):
                weights[j] -= (learning_rate / batch_m) * gradients[j]
        
        # Afficher le coût périodiquement
        if verbose and (epoch + 1) % 10 == 0:
            cost = compute_cost(X, y_binary, weights)
            print(f"  Époque {epoch + 1}/{epochs}, Coût: {cost:.6f}")
    
    return weights


def train_one_vs_all(X, y, houses, learning_rate=0.1, epochs=100, batch_size=32):
    """Entraîne un classifieur one-vs-all pour chaque maison avec Mini-Batch GD"""
    models = {}
    
    for house in houses:
        print(f"\nEntraînement du modèle pour {house} (Mini-Batch GD)...")
        
        # Créer les labels binaires (1 si house, 0 sinon)
        y_binary = [1.0 if label == house else 0.0 for label in y]
        
        # Entraîner le modèle avec Mini-Batch GD
        weights = minibatch_gradient_descent(X, y_binary, learning_rate, epochs, batch_size)
        
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
    if len(sys.argv) < 2:
        print("Usage: python logreg_train_minibatch.py <dataset_train.csv> [batch_size]", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    # Features sélectionnées (basées sur l'analyse du pair_plot)
    selected_features = [
        'Astronomy',
        'Herbology',
        'Defense Against the Dark Arts',
        'Ancient Runes',
        'Charms'
    ]
    
    print("="*70)
    print("ENTRAÎNEMENT AVEC MINI-BATCH GRADIENT DESCENT")
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
    learning_rate = 0.1
    epochs = 100
    
    num_batches = ft_ceil(len(X) / batch_size)
    
    print(f"\nParamètres d'entraînement:")
    print(f"  - Méthode: Mini-Batch Gradient Descent")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Époques: {epochs}")
    print(f"  - Taille des mini-batches: {batch_size}")
    print(f"  - Nombre de batches par époque: {num_batches}")
    print(f"  - Mises à jour par époque: {num_batches}")
    
    # Fixer la seed pour la reproductibilité
    random.seed(42)
    
    models = train_one_vs_all(X_normalized, y, houses, learning_rate, epochs, batch_size)
    
    # Sauvegarder les poids
    save_weights(models, means, stds, selected_features, 'weights_minibatch.json')
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*70)


if __name__ == "__main__":
    main()

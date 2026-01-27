#!/usr/bin/env python3
import sys

from utils import (
    ft_length, read_csv, normalize_features, predict_probability,
    compute_cost, extract_features_and_labels, save_weights
)


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

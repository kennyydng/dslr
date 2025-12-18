#!/usr/bin/env python3
"""
logreg_predict - Prédiction des maisons de Poudlard
en utilisant les modèles de régression logistique entraînés
"""

import sys
import csv
import json
import math
from collections import Counter


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


def load_weights(filepath):
    """Charge les poids et paramètres depuis le fichier JSON"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filepath}' n'existe pas.", file=sys.stderr)
        print("Veuillez d'abord entraîner le modèle avec logreg_train.py", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}", file=sys.stderr)
        sys.exit(1)


def sigmoid(z):
    """Fonction sigmoïde"""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(X, weights):
    """Calcule la probabilité avec la régression logistique"""
    z = weights[0]  # Biais
    for i in range(len(X)):
        z += weights[i + 1] * X[i]
    return sigmoid(z)


def normalize_features(features, means, stds):
    """Normalise les features avec les paramètres d'entraînement"""
    normalized = []
    for i in range(len(features)):
        normalized.append((features[i] - means[i]) / stds[i])
    return normalized


def predict_house(features, models, means, stds):
    """Prédit la maison en utilisant tous les modèles (one-vs-all)"""
    # Normaliser les features
    features_normalized = normalize_features(features, means, stds)
    
    # Calculer les probabilités pour chaque maison
    probabilities = {}
    for house, weights in models.items():
        prob = predict_probability(features_normalized, weights)
        probabilities[house] = prob
    
    # Retourner la maison avec la probabilité la plus élevée
    predicted_house = max(probabilities, key=probabilities.get)
    return predicted_house


def save_predictions(predictions, output_file='houses.csv'):
    """Sauvegarde les prédictions dans un fichier CSV"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Hogwarts House'])
        for index, house in predictions:
            writer.writerow([index, house])
    
    print(f"✓ Prédictions sauvegardées dans '{output_file}'")


def visualize_predictions(predictions_list, dataset_file=None):
    """Visualise la répartition des prédictions"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Erreur: matplotlib n'est pas installé. Impossible de visualiser.", file=sys.stderr)
        return
    
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    colors = ['#d32f2f', '#43a047', '#1976d2', '#fdd835']
    
    # Extraire les prédictions
    predictions = [house for _, house in predictions_list]
    
    # Compter les prédictions
    pred_counts = Counter(predictions)
    pred_values = [pred_counts.get(house, 0) for house in houses]
    
    # Charger les vraies valeurs si disponibles
    true_labels = None
    if dataset_file:
        try:
            with open(dataset_file, 'r') as f:
                reader = csv.DictReader(f)
                true_labels = [row.get('Hogwarts House') for row in reader 
                             if row.get('Hogwarts House')]
        except:
            true_labels = None
    
    if true_labels:
        # Si on a les vraies valeurs, les afficher aussi
        true_counts = Counter(true_labels)
        true_values = [true_counts.get(house, 0) for house in houses]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Graphique 1: Prédictions
        bars1 = ax1.bar(houses, pred_values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Prédictions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Nombre d\'étudiants', fontsize=12)
        ax1.set_xlabel('Maison', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Graphique 2: Vraies valeurs
        bars2 = ax2.bar(houses, true_values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('Vraies Valeurs', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Nombre d\'étudiants', fontsize=12)
        ax2.set_xlabel('Maison', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Graphique 3: Comparaison
        x = range(len(houses))
        width = 0.35
        
        bars3a = ax3.bar([i - width/2 for i in x], true_values, width, 
                        label='Vraies valeurs', color=colors, alpha=0.7, 
                        edgecolor='black', linewidth=1.5)
        bars3b = ax3.bar([i + width/2 for i in x], pred_values, width,
                        label='Prédictions', color=colors, alpha=0.4,
                        edgecolor='black', linewidth=1.5, hatch='//')
        
        ax3.set_title('Comparaison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Nombre d\'étudiants', fontsize=12)
        ax3.set_xlabel('Maison', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(houses)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Répartition des Maisons de Poudlard', 
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        # Seulement les prédictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Graphique en barres
        bars = ax1.bar(houses, pred_values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('Répartition des Prédictions', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Nombre d\'étudiants', fontsize=12)
        ax1.set_xlabel('Maison', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Graphique en camembert
        ax2.pie(pred_values, labels=houses, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
               wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        ax2.set_title('Proportion par Maison', fontsize=14, fontweight='bold')
        
        plt.suptitle('Répartition des Maisons Prédites - Poudlard', 
                    fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques
    print("\n" + "="*70)
    print("STATISTIQUES DE RÉPARTITION")
    print("="*70)
    
    if true_labels:
        print("\nPRÉDICTIONS:")
    
    total_pred = sum(pred_values)
    for house, count in zip(houses, pred_values):
        percentage = (count / total_pred * 100) if total_pred > 0 else 0
        print(f"  {house:15s} : {count:4d} étudiants ({percentage:5.2f}%)")
    print(f"  {'TOTAL':15s} : {total_pred:4d} étudiants")
    
    if true_labels:
        print("\nVRAIES VALEURS:")
        total_true = sum(true_values)
        for house, count in zip(houses, true_values):
            percentage = (count / total_true * 100) if total_true > 0 else 0
            print(f"  {house:15s} : {count:4d} étudiants ({percentage:5.2f}%)")
        print(f"  {'TOTAL':15s} : {total_true:4d} étudiants")
        
        # Calculer la précision si possible
        if len(predictions) == len(true_labels):
            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            accuracy = (correct / len(predictions) * 100)
            print(f"\nPRÉCISION GLOBALE: {accuracy:.2f}% ({correct}/{len(predictions)})")
    
    print("="*70 + "\n")


def main():
    """Fonction principale"""
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset_test.csv> <weights.json>", file=sys.stderr)
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    weights_file = sys.argv[2]
    
    print("="*70)
    print("PRÉDICTION DES MAISONS - RÉGRESSION LOGISTIQUE")
    print("="*70)
    
    # Charger les poids
    print(f"\nChargement des poids: {weights_file}")
    weights_data = load_weights(weights_file)
    
    selected_features = weights_data['features']
    means = weights_data['normalization']['means']
    stds = weights_data['normalization']['stds']
    models = weights_data['models']
    
    print(f"  → Features: {', '.join(selected_features)}")
    print(f"  → Modèles chargés: {', '.join(models.keys())}")
    
    # Lire le dataset de test
    print(f"\nChargement du dataset: {dataset_file}")
    data = read_csv(dataset_file)
    print(f"  → {len(data)} lignes chargées")
    
    # Faire les prédictions
    print("\nPrédiction en cours...")
    predictions = []
    valid_predictions = 0
    
    for row in data:
        index = row.get('Index')
        
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
            # Prédire la maison
            house = predict_house(features, models, means, stds)
            predictions.append((index, house))
            valid_predictions += 1
        else:
            # Si des valeurs manquent, prédire "Unknown" ou la maison la plus probable par défaut
            predictions.append((index, "Gryffindor"))  # Défaut
    
    print(f"  → {valid_predictions}/{len(data)} prédictions valides")
    
    # Sauvegarder les résultats
    print("\nSauvegarde des prédictions...")
    save_predictions(predictions)
    
    # Afficher un échantillon des prédictions
    print("\nÉchantillon des prédictions:")
    print(f"{'Index':<10} {'Maison Prédite'}")
    print("-" * 30)
    for i in range(min(10, len(predictions))):
        index, house = predictions[i]
        print(f"{index:<10} {house}")
    if len(predictions) > 10:
        print("...")
    
    print("\n" + "="*70)
    print("PRÉDICTION TERMINÉE")
    print("="*70)
    
    # Visualiser automatiquement
    print("\nAffichage de la visualisation...")
    visualize_predictions(predictions, dataset_file)


if __name__ == "__main__":
    main()

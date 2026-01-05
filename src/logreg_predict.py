#!/usr/bin/env python3
"""
logreg_predict - Prédiction des maisons de Poudlard
en utilisant les modèles de régression logistique entraînés
"""

import sys
import csv
import json
import math

from utils import clamp, ft_length, argmax_dict, count_occurrences, ft_sum


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
    z = clamp(z, -500, 500)
    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(X, weights):
    """Calcule la probabilité avec la régression logistique"""
    z = weights[0]  # Biais
    i = 0
    n = ft_length(X)
    while i < n:
        z += weights[i + 1] * X[i]
        i += 1
    return sigmoid(z)


def normalize_features(features, means, stds):
    """Normalise les features avec les paramètres d'entraînement"""
    normalized = []
    i = 0
    n = ft_length(features)
    while i < n:
        normalized.append((features[i] - means[i]) / stds[i])
        i += 1
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
    predicted_house = argmax_dict(probabilities)
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
    predictions = []
    for _, house in predictions_list:
        predictions.append(house)
    
    # Compter les prédictions
    pred_counts = count_occurrences(predictions)
    pred_values = []
    for house in houses:
        pred_values.append(pred_counts.get(house, 0))
    
    # Charger les vraies valeurs si disponibles
    true_labels = None
    if dataset_file:
        try:
            with open(dataset_file, 'r') as f:
                reader = csv.DictReader(f)
                labels = []
                for row in reader:
                    label = row.get('Hogwarts House')
                    if label:
                        labels.append(label)
                true_labels = labels
        except:
            true_labels = None
    
    if true_labels:
        # Si on a les vraies valeurs, les afficher aussi
        true_counts = count_occurrences(true_labels)
        true_values = []
        for house in houses:
            true_values.append(true_counts.get(house, 0))
        
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
        x = range(ft_length(houses))
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
    
    total_pred = ft_sum(pred_values)
    i = 0
    n = ft_length(houses)
    while i < n:
        house = houses[i]
        count = pred_values[i]
        percentage = (count / total_pred * 100) if total_pred > 0 else 0
        print(f"  {house:15s} : {count:4d} étudiants ({percentage:5.2f}%)")
        i += 1
    print(f"  {'TOTAL':15s} : {total_pred:4d} étudiants")
    
    if true_labels:
        print("\nVRAIES VALEURS:")
        total_true = ft_sum(true_values)
        i = 0
        n = ft_length(houses)
        while i < n:
            house = houses[i]
            count = true_values[i]
            percentage = (count / total_true * 100) if total_true > 0 else 0
            print(f"  {house:15s} : {count:4d} étudiants ({percentage:5.2f}%)")
            i += 1
        print(f"  {'TOTAL':15s} : {total_true:4d} étudiants")
        
        # Calculer la précision si possible
        pred_n = ft_length(predictions)
        true_n = ft_length(true_labels)
        if pred_n == true_n:
            correct = 0
            i = 0
            while i < pred_n:
                if predictions[i] == true_labels[i]:
                    correct += 1
                i += 1
            accuracy = (correct / pred_n * 100) if pred_n > 0 else 0
            print(f"\nPRÉCISION GLOBALE: {accuracy:.2f}% ({correct}/{pred_n})")
    
    print("="*70 + "\n")


def main():
    """Fonction principale"""
    if ft_length(sys.argv) != 3:
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
    data_n = ft_length(data)
    print(f"  → {data_n} lignes chargées")
    
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
    
    print(f"  → {valid_predictions}/{data_n} prédictions valides")
    
    # Sauvegarder les résultats
    print("\nSauvegarde des prédictions...")
    save_predictions(predictions)
    
    # Afficher un échantillon des prédictions
    print("\nÉchantillon des prédictions:")
    print(f"{'Index':<10} {'Maison Prédite'}")
    print("-" * 30)
    pred_n = ft_length(predictions)
    limit = 10
    if pred_n < limit:
        limit = pred_n
    i = 0
    while i < limit:
        index, house = predictions[i]
        print(f"{index:<10} {house}")
        i += 1
    if pred_n > 10:
        print("...")
    
    print("\n" + "="*70)
    print("PRÉDICTION TERMINÉE")
    print("="*70)
    
    # Visualiser automatiquement
    print("\nAffichage de la visualisation...")
    visualize_predictions(predictions, dataset_file)


if __name__ == "__main__":
    main()

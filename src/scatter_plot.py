#!/usr/bin/env python3
"""
Script scatter_plot - Affiche un scatter plot pour trouver
les deux features les plus similaires
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


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


def parse_float(value):
    """Convertit une valeur en float, retourne None si impossible"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def extract_features(data):
    """Extrait les features numériques"""
    features = [
        'Arithmancy', 'Astronomy', 'Herbology', 
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]
    
    feature_data = {feature: [] for feature in features}
    
    for row in data:
        for feature in features:
            value = parse_float(row.get(feature))
            if value is not None:
                feature_data[feature].append(value)
            else:
                feature_data[feature].append(np.nan)
    
    return feature_data


def calculate_correlation(x, y):
    """Calcule le coefficient de corrélation de Pearson"""
    # Filtrer les valeurs NaN
    valid_indices = ~(np.isnan(x) | np.isnan(y))
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]
    
    if len(x_valid) < 2:
        return 0.0
    
    return np.corrcoef(x_valid, y_valid)[0, 1]


def find_most_correlated_features(feature_data):
    """Trouve les deux features les plus corrélées"""
    features = list(feature_data.keys())
    correlations = {}
    
    for feat1, feat2 in combinations(features, 2):
        x = np.array(feature_data[feat1])
        y = np.array(feature_data[feat2])
        
        corr = calculate_correlation(x, y)
        correlations[(feat1, feat2)] = abs(corr)
    
    # Trouver la paire avec la plus forte corrélation
    most_similar = max(correlations, key=correlations.get)
    
    return most_similar, correlations


def plot_scatter(feature_data, most_similar, correlations):
    """Affiche les scatter plots"""
    features = list(feature_data.keys())
    
    # Créer une figure pour les paires les plus corrélées
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Scatter Plots - Recherche de features similaires', fontsize=16)
    
    # Trier par corrélation
    sorted_pairs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:12]
    
    for idx, ((feat1, feat2), corr) in enumerate(sorted_pairs, 1):
        ax = plt.subplot(3, 4, idx)
        
        x = np.array(feature_data[feat1])
        y = np.array(feature_data[feat2])
        
        # Filtrer les NaN
        valid = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]
        
        # Scatter plot
        ax.scatter(x_valid, y_valid, alpha=0.5, s=10)
        
        # Ligne de régression
        if len(x_valid) > 1:
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            ax.plot(x_valid, p(x_valid), "r--", alpha=0.8, linewidth=2)
        
        # Mettre en évidence la paire la plus similaire
        if (feat1, feat2) == most_similar:
            title = f'{feat1[:15]}...\nvs\n{feat2[:15]}...\n★ r={corr:.3f} ★'
            ax.set_title(title, fontweight='bold', fontsize=9, color='red')
            ax.patch.set_edgecolor('red')
            ax.patch.set_linewidth(3)
        else:
            title = f'{feat1[:15]}...\nvs\n{feat2[:15]}...\nr={corr:.3f}'
            ax.set_title(title, fontsize=8)
        
        ax.set_xlabel(feat1[:20], fontsize=7)
        ax.set_ylabel(feat2[:20], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Lire les données
    data = read_csv(filepath)
    
    # Extraire les features
    feature_data = extract_features(data)
    
    # Trouver les features les plus similaires
    most_similar, correlations = find_most_correlated_features(feature_data)
    
    # Afficher les résultats
    print("\n" + "="*70)
    print("ANALYSE DE SIMILARITÉ ENTRE FEATURES")
    print("="*70)
    print(f"\nLes deux features les plus similaires:")
    print(f"  → {most_similar[0]}")
    print(f"  → {most_similar[1]}")
    print(f"  → Corrélation: {correlations[most_similar]:.6f}")
    
    print("\nTop 10 des paires les plus corrélées:")
    for i, ((feat1, feat2), corr) in enumerate(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"  {i}. {feat1:30s} <-> {feat2:30s} : r={corr:.6f}")
    print("="*70 + "\n")
    
    # Afficher les scatter plots
    plot_scatter(feature_data, most_similar, correlations)


if __name__ == "__main__":
    main()

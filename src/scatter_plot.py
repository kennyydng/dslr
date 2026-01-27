#!/usr/bin/env python3
"""
Script scatter_plot - Affiche un scatter plot pour trouver
les deux features les plus similaires
"""
import sys
import matplotlib.pyplot as plt
from itertools import combinations

from math_utils import (
    argmax_dict,
    ft_length,
    sort_pairs_by_value,
    parse_float,
    read_csv
)
from ml_utils import pearson_corr

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
                feature_data[feature].append(None)
    
    return feature_data


def find_most_correlated_features(feature_data):
    """Trouve les deux features les plus corrélées"""
    features = list(feature_data.keys())
    correlations = {}
    
    for feat1, feat2 in combinations(features, 2):
        x = feature_data[feat1]
        y = feature_data[feat2]
        
        corr = pearson_corr(x, y)
        correlations[(feat1, feat2)] = abs(corr)
    
    # Trouver la paire avec la plus forte corrélation
    most_similar = argmax_dict(correlations)
    
    return most_similar, correlations


def plot_scatter(feature_data, most_similar, correlations):
    """Affiche les scatter plots"""
    features = list(feature_data.keys())
    
    # Créer une figure pour les paires les plus corrélées
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Scatter Plots - Recherche de features similaires', fontsize=16)
    
    # Trier par corrélation
    sorted_pairs = sort_pairs_by_value(list(correlations.items()), reverse=True)
    limit = 12
    if ft_length(sorted_pairs) < limit:
        limit = ft_length(sorted_pairs)
    
    idx = 1
    pair_i = 0
    while pair_i < limit:
        (feat1, feat2), corr = sorted_pairs[pair_i]
        ax = plt.subplot(3, 4, idx)

        x_raw = feature_data[feat1]
        y_raw = feature_data[feat2]

        x_valid = []
        y_valid = []
        i = 0
        n = ft_length(x_raw)
        while i < n:
            x = x_raw[i]
            y = y_raw[i]
            if x is not None and y is not None:
                x_valid.append(x)
                y_valid.append(y)
            i += 1
        
        ax.scatter(x_valid, y_valid, alpha=0.5, s=10)
        
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

        idx += 1
        pair_i += 1
    
    plt.tight_layout()
    plt.show()


def main():
    if ft_length(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    data = read_csv(filepath)
    feature_data = extract_features(data)
    most_similar, correlations = find_most_correlated_features(feature_data)
    
    print("\n" + "="*70)
    print("ANALYSE DE SIMILARITÉ ENTRE FEATURES")
    print("="*70)
    print(f"\nLes deux features les plus similaires:")
    print(f"  → {most_similar[0]}")
    print(f"  → {most_similar[1]}")
    print(f"  → Corrélation: {correlations[most_similar]:.6f}")
    print("\nTop 10 des paires les plus corrélées:")
    ranked = sort_pairs_by_value(list(correlations.items()), reverse=True)
    limit = 10
    i = 1
    idx = 0

    if ft_length(ranked) < limit:
        limit = ft_length(ranked)

    while idx < limit:
        (feat1, feat2), corr = ranked[idx]
        print(f"  {i}. {feat1:30s} <-> {feat2:30s} : r={corr:.6f}")
        i += 1
        idx += 1
    print("="*70 + "\n")
    
    plot_scatter(feature_data, most_similar, correlations)


if __name__ == "__main__":
    main()

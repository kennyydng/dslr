#!/usr/bin/env python3
"""pair_plot - Pair plot + scoring de features (version manuelle).

Cette version évite pandas/numpy/seaborn et n'utilise pas de fonctions qui font les
statistiques "toutes seules" (mean/var/corr/describe). Les calculs sont faits via
les helpers manuels dans utils.py.
"""

import sys
import csv
import matplotlib.pyplot as plt

from utils import (
    ft_length,
    mean,
    variance,
    pearson_corr,
    sort_pairs_by_value,
    ft_min,
    ft_max,
)


HOUSES = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
NUMERIC_FEATURES = [
    'Arithmancy', 'Astronomy', 'Herbology',
    'Defense Against the Dark Arts', 'Divination',
    'Muggle Studies', 'Ancient Runes', 'History of Magic',
    'Transfiguration', 'Potions', 'Care of Magical Creatures',
    'Charms', 'Flying'
]


def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def read_csv(filepath):
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filepath}' n'existe pas.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}", file=sys.stderr)
        sys.exit(1)


def build_columns(data, features):
    columns = {}
    for feat in features:
        columns[feat] = []

    houses = []
    for row in data:
        houses.append(row.get('Hogwarts House'))
        for feat in features:
            columns[feat].append(parse_float(row.get(feat)))

    return columns, houses


def analyze_feature_importance(data, features):
    """Score = variance(means_by_house) / mean(variances_by_house)."""

    feature_scores = {}

    for feat in features:
        per_house_values = {}
        for h in HOUSES:
            per_house_values[h] = []

        for row in data:
            house = row.get('Hogwarts House')
            if house not in HOUSES:
                continue
            val = parse_float(row.get(feat))
            if val is None:
                continue
            per_house_values[house].append(val)

        house_means = []
        house_vars = []
        for h in HOUSES:
            vals = per_house_values[h]
            if ft_length(vals) == 0:
                continue
            m = mean(vals)
            v = variance(vals, mean_val=m)
            house_means.append(m)
            house_vars.append(v)

        intra = mean(house_vars) if ft_length(house_vars) > 0 else 0.0
        inter = variance(house_means) if ft_length(house_means) > 0 else 0.0

        score = (inter / intra) if intra > 0 else 0.0
        feature_scores[feat] = score

    return feature_scores


def plot_full_correlation_matrix(columns, features):
    """Correlation matrix computed manually, displayed with matplotlib."""

    n = ft_length(features)
    matrix = []
    i = 0
    while i < n:
        row = []
        j = 0
        while j < n:
            f1 = features[i]
            f2 = features[j]
            r = pearson_corr(columns[f1], columns[f2])
            row.append(r)
            j += 1
        matrix.append(row)
        i += 1

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title('Matrice de Corrélation - Toutes les Features', fontsize=16, pad=20)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(features, rotation=90, fontsize=8)
    ax.set_yticklabels(features, fontsize=8)

    # Annotate values
    i = 0
    while i < n:
        j = 0
        while j < n:
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', fontsize=6)
            j += 1
        i += 1

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_pair_plot(data, top_features):
    """Scatter-matrix for selected features (manual)."""

    colors = {
        'Gryffindor': '#740001',
        'Slytherin': '#1a472a',
        'Ravenclaw': '#0e1a40',
        'Hufflepuff': '#ecb939',
    }

    columns, houses = build_columns(data, top_features)
    n = ft_length(top_features)

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    fig.suptitle('Pair Plot - Top Features (manual)', y=1.02, fontsize=16)

    i = 0
    while i < n:
        j = 0
        while j < n:
            ax = axes[i][j] if n > 1 else axes

            feat_y = top_features[i]
            feat_x = top_features[j]

            if i == j:
                # Diagonal: histogram of the feature
                vals = []
                for v in columns[feat_x]:
                    if v is not None:
                        vals.append(v)
                if ft_length(vals) > 0:
                    ax.hist(vals, bins=20, color='gray', alpha=0.7, edgecolor='black', linewidth=0.3)
                ax.set_xlabel(feat_x, fontsize=8)
                ax.set_ylabel('Freq', fontsize=8)
            else:
                # Off-diagonal: scatter by house
                for house in HOUSES:
                    xs = []
                    ys = []
                    k = 0
                    m = ft_length(houses)
                    while k < m:
                        if houses[k] == house:
                            x = columns[feat_x][k]
                            y = columns[feat_y][k]
                            if x is not None and y is not None:
                                xs.append(x)
                                ys.append(y)
                        k += 1
                    if ft_length(xs) > 0:
                        ax.scatter(xs, ys, s=8, alpha=0.5, c=colors[house], label=house)

                if i == 0 and j == n - 1:
                    ax.legend(fontsize=7, loc='best')

                ax.set_xlabel(feat_x, fontsize=8)
                ax.set_ylabel(feat_y, fontsize=8)

            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
            j += 1
        i += 1

    plt.tight_layout()
    plt.show()


def main():
    if ft_length(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]
    data = read_csv(filepath)

    scores = analyze_feature_importance(data, NUMERIC_FEATURES)
    ranked = sort_pairs_by_value(list(scores.items()), reverse=True)

    top_n = 6
    if ft_length(ranked) < top_n:
        top_n = ft_length(ranked)

    top_features = []
    print("\n" + "=" * 70)
    print("FEATURES SÉLECTIONNÉES POUR LE PAIR PLOT")
    print("=" * 70)
    idx = 0
    i = 1
    while idx < top_n:
        feat, score = ranked[idx]
        top_features.append(feat)
        print(f"  {i}. {feat:40s} : score = {score:.6f}")
        idx += 1
        i += 1
    print("=" * 70 + "\n")

    # Correlation matrix on all features
    columns, _ = build_columns(data, NUMERIC_FEATURES)
    print("Affichage de la matrice de corrélation...")
    plot_full_correlation_matrix(columns, NUMERIC_FEATURES)

    # Pair plot on selected features
    print("Création du pair plot (manual)...")
    plot_pair_plot(data, top_features)


if __name__ == "__main__":
    main()

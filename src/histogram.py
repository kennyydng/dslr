#!/usr/bin/env python3
import sys
import csv
import matplotlib.pyplot as plt

from utils import (
    argmin_dict,
    ft_length,
    ft_max,
    ft_min,
    ft_mean,
    sort_pairs_by_value,
    ft_variance,
)


def read_csv(filepath):
    """Lit un fichier CSV et retourne les headers et les données"""
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


def get_course_scores_by_house(data):
    """Extrait les scores de chaque cours par maison"""
    courses = [
        'Arithmancy', 'Astronomy', 'Herbology', 
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]
    
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    
    # Structure: {course: {house: [scores]}}
    course_data = {course: {house: [] for house in houses} for course in courses}
    
    for row in data:
        house = row.get('Hogwarts House')
        if house not in houses:
            continue
            
        for course in courses:
            score = parse_float(row.get(course))
            if score is not None:
                course_data[course][house].append(score)
    
    return course_data


def calculate_homogeneity(course_data):
    """
    Calcule l'homogénéité de la distribution pour chaque cours.
    Plus la variance des moyennes entre maisons est faible, plus c'est homogène.
    """
    homogeneity_scores = {}
    
    for course, houses_scores in course_data.items():
        # Calculer la moyenne de chaque maison
        means = []
        for house, scores in houses_scores.items():
            if scores:
                means.append(ft_mean(scores))
        
        if means:
            # Variance des moyennes (plus c'est faible, plus c'est homogène)
            homogeneity_scores[course] = ft_variance(means)
    
    return homogeneity_scores


def plot_histograms(course_data, most_homogeneous):
    """Affiche les histogrammes des cours"""
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    colors = ['#740001', '#1a472a', '#0e1a40', '#ecb939']
    courses = list(course_data.keys())
    n_courses = ft_length(courses)
    
    # Disposition en grille
    n_cols = 4
    n_rows = (n_courses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    fig.suptitle('Distribution of Scores by Course and House', fontsize=16, y=0.995)
    
    # Aplatir les axes pour faciliter l'itération
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, course in enumerate(courses):
        ax = axes_flat[idx]
        
        all_scores = []
        for scores in course_data[course].values():
            all_scores.extend(scores)
        
        if not all_scores:
            ax.set_visible(False)
            continue
        
        min_s = ft_min(all_scores)
        max_s = ft_max(all_scores)
        n_bins = 30

        if max_s == min_s:
            max_s = min_s + 1.0
        step = (max_s - min_s) / n_bins
        bins = []

        i = 0
        while i <= n_bins:
            bins.append(min_s + step * i)
            i += 1
        
        for house, color in zip(houses, colors):
            scores = course_data[course][house]
            if scores:
                ax.hist(scores, bins=bins, alpha=0.5, label=house, color=color, edgecolor='black', linewidth=0.5)
        
        if course == most_homogeneous:
            ax.set_title(f'{course}\n★ MOST HOMOGENEOUS ★', fontweight='bold', fontsize=10, color='red')
            ax.patch.set_edgecolor('red')
            ax.patch.set_linewidth(3)
        else:
            ax.set_title(course, fontsize=9)
        
        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Fréquence', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(ft_length(courses), ft_length(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale"""
    if ft_length(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    data = read_csv(filepath)
    course_data = get_course_scores_by_house(data)
    homogeneity = calculate_homogeneity(course_data)
    most_homogeneous = argmin_dict(homogeneity)
    
    print("\n" + "="*70)
    print("ANALYSE DE L'HOMOGÉNÉITÉ DES DISTRIBUTIONS PAR COURS")
    print("="*70)
    print(f"\nLe cours avec la distribution la plus homogène entre les maisons:")
    print(f"  → {most_homogeneous}")
    print(f"  → Variance des moyennes: {homogeneity[most_homogeneous]:.6f}")
    print("\nClassement par homogénéité (variance des moyennes):")
    ranked = sort_pairs_by_value(list(homogeneity.items()), reverse=False)
    i = 1
    limit = 5

    if ft_length(ranked) < limit:
        limit = ft_length(ranked)

    idx = 0

    while idx < limit:
        course, var = ranked[idx]
        print(f"  {i}. {course:30s} : {var:.6f}")
        i += 1
        idx += 1
    print("="*70 + "\n")
    
    plot_histograms(course_data, most_homogeneous)


if __name__ == "__main__":
    main()

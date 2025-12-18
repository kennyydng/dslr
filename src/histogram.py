#!/usr/bin/env python3
"""
Script histogram - Affiche un histogramme pour trouver le cours avec
la distribution de scores la plus homogène entre les quatre maisons
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


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
    # Liste des cours à analyser
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
                means.append(np.mean(scores))
        
        if means:
            # Variance des moyennes (plus c'est faible, plus c'est homogène)
            variance = np.var(means)
            homogeneity_scores[course] = variance
    
    return homogeneity_scores


def plot_histograms(course_data, most_homogeneous):
    """Affiche les histogrammes des cours"""
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    colors = ['#740001', '#1a472a', '#0e1a40', '#ecb939']
    
    # Créer une figure avec tous les cours
    courses = list(course_data.keys())
    n_courses = len(courses)
    
    # Disposition en grille
    n_cols = 4
    n_rows = (n_courses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    fig.suptitle('Distribution des scores par cours et par maison', fontsize=16, y=0.995)
    
    # Aplatir les axes pour faciliter l'itération
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, course in enumerate(courses):
        ax = axes_flat[idx]
        
        # Collecter toutes les données pour déterminer les bins
        all_scores = []
        for scores in course_data[course].values():
            all_scores.extend(scores)
        
        if not all_scores:
            ax.set_visible(False)
            continue
        
        # Créer les bins
        bins = np.linspace(min(all_scores), max(all_scores), 30)
        
        # Tracer l'histogramme pour chaque maison
        for house, color in zip(houses, colors):
            scores = course_data[course][house]
            if scores:
                ax.hist(scores, bins=bins, alpha=0.5, label=house, color=color, edgecolor='black', linewidth=0.5)
        
        # Mettre en évidence le cours le plus homogène
        if course == most_homogeneous:
            ax.set_title(f'{course}\n★ PLUS HOMOGÈNE ★', fontweight='bold', fontsize=10, color='red')
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
    for idx in range(len(courses), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Lire les données
    data = read_csv(filepath)
    
    # Extraire les scores par cours et par maison
    course_data = get_course_scores_by_house(data)
    
    # Calculer l'homogénéité
    homogeneity = calculate_homogeneity(course_data)
    
    # Trouver le cours le plus homogène (variance la plus faible)
    most_homogeneous = min(homogeneity, key=homogeneity.get)
    
    print("\n" + "="*70)
    print("ANALYSE DE L'HOMOGÉNÉITÉ DES DISTRIBUTIONS PAR COURS")
    print("="*70)
    print(f"\nLe cours avec la distribution la plus homogène entre les maisons:")
    print(f"  → {most_homogeneous}")
    print(f"  → Variance des moyennes: {homogeneity[most_homogeneous]:.6f}")
    print("\nClassement par homogénéité (variance des moyennes):")
    for i, (course, var) in enumerate(sorted(homogeneity.items(), key=lambda x: x[1])[:5], 1):
        print(f"  {i}. {course:30s} : {var:.6f}")
    print("="*70 + "\n")
    
    # Afficher les histogrammes
    plot_histograms(course_data, most_homogeneous)


if __name__ == "__main__":
    main()

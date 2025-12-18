#!/usr/bin/env python3
"""
Script pair_plot - Affiche un pair plot (scatter plot matrix)
pour visualiser les relations entre toutes les features
et aider à sélectionner les features pour la régression logistique
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def read_csv(filepath):
    """Lit un fichier CSV avec pandas"""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filepath}' n'existe pas.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}", file=sys.stderr)
        sys.exit(1)


def select_features(df):
    """Sélectionne les features numériques pertinentes"""
    numeric_features = [
        'Arithmancy', 'Astronomy', 'Herbology', 
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]
    
    # Créer un DataFrame avec seulement les features numériques et la maison
    df_filtered = df[['Hogwarts House'] + numeric_features].copy()
    
    return df_filtered, numeric_features


def analyze_feature_importance(df, numeric_features):
    """
    Analyse l'importance des features pour la classification
    en calculant la séparation entre les maisons
    """
    houses = df['Hogwarts House'].unique()
    houses = [h for h in houses if pd.notna(h)]
    
    feature_scores = {}
    
    for feature in numeric_features:
        # Calculer la variance inter-maisons vs intra-maisons
        # Plus le ratio est élevé, plus la feature est discriminante
        
        # Variance totale
        total_var = df[feature].var()
        
        # Variance intra-maisons (moyenne des variances par maison)
        intra_var = np.mean([df[df['Hogwarts House'] == house][feature].var() 
                             for house in houses])
        
        # Variance inter-maisons
        house_means = [df[df['Hogwarts House'] == house][feature].mean() 
                      for house in houses]
        inter_var = np.var(house_means)
        
        # Score de séparation (Fisher's criterion)
        if intra_var > 0:
            score = inter_var / intra_var
        else:
            score = 0
        
        feature_scores[feature] = score
    
    return feature_scores


def plot_pair_plot(df_filtered, feature_scores):
    """Crée un pair plot avec seaborn"""
    # Réduire le nombre de features pour la lisibilité
    # Sélectionner les top features les plus discriminantes
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:6]
    top_feature_names = [f[0] for f in top_features]
    
    print("\n" + "="*70)
    print("FEATURES SÉLECTIONNÉES POUR LE PAIR PLOT")
    print("="*70)
    for i, (feature, score) in enumerate(top_features, 1):
        print(f"  {i}. {feature:40s} : score = {score:.6f}")
    print("="*70 + "\n")
    
    # Créer le pair plot avec les top features
    df_top = df_filtered[['Hogwarts House'] + top_feature_names].copy()
    
    # Configurer le style
    sns.set_style("whitegrid")
    
    # Créer le pair plot
    print("Création du pair plot (cela peut prendre quelques secondes)...")
    g = sns.pairplot(
        df_top, 
        hue='Hogwarts House',
        palette={'Gryffindor': '#740001', 'Slytherin': '#1a472a', 
                 'Ravenclaw': '#0e1a40', 'Hufflepuff': '#ecb939'},
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 20},
        diag_kws={'alpha': 0.7, 'linewidth': 2}
    )
    
    g.fig.suptitle('Pair Plot - Top 6 Features les plus discriminantes', 
                   y=1.01, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def plot_full_correlation_matrix(df, numeric_features):
    """Affiche la matrice de corrélation complète"""
    # Calculer la matrice de corrélation
    corr_matrix = df[numeric_features].corr()
    
    # Créer la heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Matrice de Corrélation - Toutes les Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Lire les données
    df = read_csv(filepath)
    
    # Sélectionner les features
    df_filtered, numeric_features = select_features(df)
    
    # Analyser l'importance des features
    feature_scores = analyze_feature_importance(df_filtered, numeric_features)
    
    # Afficher les résultats de l'analyse
    print("\n" + "="*70)
    print("RECOMMANDATIONS POUR LA RÉGRESSION LOGISTIQUE")
    print("="*70)
    print("\nFeatures classées par pouvoir discriminant (variance inter/intra maisons):")
    for i, (feature, score) in enumerate(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), 1):
        recommendation = "★ FORTEMENT RECOMMANDÉE" if i <= 5 else "Recommandée" if i <= 8 else "Optionnelle"
        print(f"  {i:2d}. {feature:40s} : {score:8.6f}  [{recommendation}]")
    
    print("\n" + "="*70)
    print("\nConseils pour sélectionner les features:")
    print("  1. Privilégier les features avec un score élevé")
    print("  2. Éviter les features trop corrélées entre elles")
    print("  3. Vérifier la séparation visuelle dans le pair plot")
    print("  4. Commencer avec 4-6 features puis affiner")
    print("="*70 + "\n")
    
    # Afficher la matrice de corrélation
    print("Affichage de la matrice de corrélation...")
    plot_full_correlation_matrix(df, numeric_features)
    
    # Afficher le pair plot
    plot_pair_plot(df_filtered, feature_scores)


if __name__ == "__main__":
    main()

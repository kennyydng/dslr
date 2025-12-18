#!/usr/bin/env python3
"""
Script de vérification - Compare les résultats de describe.py avec pandas
"""

import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: python verify_with_pandas.py <dataset.csv>")
    sys.exit(1)

filepath = sys.argv[1]

# Charger le dataset avec pandas
df = pd.read_csv(filepath)

# Afficher les statistiques descriptives pour les colonnes numériques
print("=" * 80)
print("STATISTIQUES AVEC PANDAS (pour vérification)")
print("=" * 80)
print(df.describe())
print("\n")
print("Nombre de valeurs manquantes par colonne :")
print(df.isnull().sum())

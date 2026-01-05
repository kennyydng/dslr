#!/usr/bin/env python3
"""
Programme describe - Analyse descriptive d'un dataset
Affiche les statistiques descriptives pour toutes les features numériques
"""

import sys
import csv

from utils import (
    ft_length,
    ft_min,
    ft_max,
    merge_sort,
    str_length,
    is_nan,
    ft_floor,
    ft_ceil,
    ft_mean,
    ft_variance,
    ft_std,
)


def parse_float(value):
    """Convertit une valeur en float, retourne None si impossible"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def read_csv(filepath):
    """Lit un fichier CSV et retourne les headers et les données"""
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = list(reader)
        return headers, data
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filepath}' n'existe pas.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}", file=sys.stderr)
        sys.exit(1)


def shorten_column_name(name):
    """Raccourcit les noms de colonnes trop longs"""
    # Garder les noms complets mais les diviser sur deux lignes si nécessaire
    split_names = {
        'Defense Against the Dark Arts': 'Defense Against\nthe Dark Arts',
        'Care of Magical Creatures': 'Care of Magical\nCreatures',
        'History of Magic': 'History of\nMagic',
        'Muggle Studies': 'Muggle\nStudies',
        'Ancient Runes': 'Ancient\nRunes'
    }
    return split_names.get(name, name)


def extract_numeric_columns(headers, data):
    """Extrait les colonnes numériques du dataset"""
    numeric_data = {}
    
    # Colonnes à exclure (identifiants, noms, etc.)
    excluded_columns = {'Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Hogwarts House'}
    
    for col_idx, header in enumerate(headers):
        # Ignorer les colonnes non pertinentes
        if header in excluded_columns:
            continue
            
        column_values = []
        
        for row in data:
            try:
                raw = row[col_idx]
            except IndexError:
                continue

            value = parse_float(raw)
            if value is not None:
                column_values.append(value)
        
        # On considère une colonne comme numérique si elle contient au moins une valeur numérique
        if column_values:
            short_name = shorten_column_name(header)
            numeric_data[short_name] = column_values
    
    return numeric_data


def count(values):
    """Compte le nombre de valeurs non-nulles"""
    return ft_length(values)


def mean(values):
    """Calcule la moyenne"""
    return ft_mean(values)


def variance(values, mean_val):
    """Calcule la variance"""
    return ft_variance(values, mean_val)


def std(values):
    """Calcule l'écart-type"""
    return ft_std(values)


def min_value(values):
    """Retourne la valeur minimale"""
    return ft_min(values)


def max_value(values):
    """Retourne la valeur maximale"""
    return ft_max(values)


def percentile(values, p):
    """Calcule le percentile p (0-100)"""
    if ft_length(values) == 0:
        return float('nan')
    
    sorted_values = merge_sort(values)
    n = ft_length(sorted_values)
    
    # Méthode linéaire d'interpolation
    k = (n - 1) * (p / 100.0)
    f = ft_floor(k)
    c = ft_ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def range_value(values):
    """Calcule l'étendue (range) = max - min"""
    if ft_length(values) == 0:
        return float('nan')
    return max_value(values) - min_value(values)



def skewness(values):
    """Calcule le coefficient d'asymétrie (skewness)"""
    if ft_length(values) < 3:
        return float('nan')
    
    n = ft_length(values)
    mean_val = mean(values)
    std_val = std(values)
    
    if std_val == 0 or is_nan(std_val):
        return float('nan')
    
    # Formule: E[((X - μ) / σ)³]
    acc = 0.0
    for x in values:
        z = (x - mean_val) / std_val
        acc += z * z * z
    return acc / n


def kurtosis(values):
    """Calcule le coefficient d'aplatissement (kurtosis)"""
    if ft_length(values) < 4:
        return float('nan')
    
    n = ft_length(values)
    mean_val = mean(values)
    std_val = std(values)
    
    if std_val == 0 or is_nan(std_val):
        return float('nan')
    
    # Formule: E[((X - μ) / σ)⁴] - 3 (excess kurtosis)
    acc = 0.0
    for x in values:
        z = (x - mean_val) / std_val
        acc += z * z * z * z
    return (acc / n) - 3


def calculate_statistics(numeric_data):
    """Calcule toutes les statistiques pour chaque colonne numérique"""
    stats = {}
    
    for column, values in numeric_data.items():
        stats[column] = {
            'Count': count(values),
            'Mean': mean(values),
            'Std': std(values),
            'Min': min_value(values),
            '25%': percentile(values, 25),
            '50%': percentile(values, 50),
            '75%': percentile(values, 75),
            'Max': max_value(values),
            'Range': range_value(values),
            'Skewness': skewness(values),
            'Kurtosis': kurtosis(values)
        }
    
    return stats


def display_statistics(stats):
    """Affiche les statistiques sous forme de tableau formaté"""
    if not stats:
        print("Aucune donnée numérique trouvée.")
        return
    
    # Définir les lignes à afficher
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Range', 'Skewness', 'Kurtosis']
    columns = list(stats.keys())
    
    # Séparer les noms de colonnes en lignes
    col_lines = []
    max_lines = 1
    for col in columns:
        lines = col.split('\n')
        col_lines.append(lines)
        lines_n = ft_length(lines)
        if lines_n > max_lines:
            max_lines = lines_n
    
    # Calculer la largeur optimale pour chaque colonne
    col_widths = {}
    for i, col in enumerate(columns):
        # Largeur minimale = longueur max des lignes du nom
        max_width = 0
        for line in col_lines[i]:
            w = str_length(line)
            if w > max_width:
                max_width = w
        
        # Vérifier la largeur nécessaire pour chaque statistique
        for stat_name in stat_names:
            value = stats[col][stat_name]
            if is_nan(value):
                width = 3  # "NaN"
            elif stat_name == 'Count':
                width = str_length(f"{value:.0f}")
            else:
                width = str_length(f"{value:.6f}")
            if width > max_width:
                max_width = width
        
        # Ajouter 2 espaces de padding
        col_widths[col] = max_width + 2
    
    first_col_width = 0
    for name in stat_names:
        w = str_length(name)
        if w > first_col_width:
            first_col_width = w
    first_col_width += 2
    
    # En-tête sur plusieurs lignes
    for line_idx in range(max_lines):
        header = ' ' * first_col_width
        for i, col in enumerate(columns):
            # Prendre la ligne correspondante ou une chaîne vide
            line_text = col_lines[i][line_idx] if line_idx < ft_length(col_lines[i]) else ''
            header += f"{line_text:>{col_widths[col]}}"
        print(header)
    
    # Lignes de statistiques
    for stat_name in stat_names:
        row = f"{stat_name:<{first_col_width}}"
        for col in columns:
            value = stats[col][stat_name]
            width = col_widths[col]
            if is_nan(value):
                row += f"{'NaN':>{width}}"
            elif stat_name == 'Count':
                row += f"{value:>{width}.0f}"
            else:
                row += f"{value:>{width}.6f}"
        print(row)


def main():
    """Fonction principale"""
    if ft_length(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Lire le fichier CSV
    headers, data = read_csv(filepath)
    
    # Extraire les colonnes numériques
    numeric_data = extract_numeric_columns(headers, data)
    
    # Calculer les statistiques
    stats = calculate_statistics(numeric_data)
    
    # Afficher les résultats
    display_statistics(stats)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Programme describe - Analyse descriptive d'un dataset
Affiche les statistiques descriptives pour toutes les features numériques
"""

import sys
import csv
import math


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
            if col_idx < len(row):
                value = parse_float(row[col_idx])
                if value is not None:
                    column_values.append(value)
        
        # On considère une colonne comme numérique si elle contient au moins une valeur numérique
        if column_values:
            short_name = shorten_column_name(header)
            numeric_data[short_name] = column_values
    
    return numeric_data


def count(values):
    """Compte le nombre de valeurs non-nulles"""
    return len(values)


def mean(values):
    """Calcule la moyenne"""
    if not values:
        return float('nan')
    return sum(values) / len(values)


def variance(values, mean_val):
    """Calcule la variance"""
    if not values:
        return float('nan')
    return sum((x - mean_val) ** 2 for x in values) / len(values)


def std(values):
    """Calcule l'écart-type"""
    if not values:
        return float('nan')
    mean_val = mean(values)
    var = variance(values, mean_val)
    return math.sqrt(var)


def min_value(values):
    """Retourne la valeur minimale"""
    if not values:
        return float('nan')
    return min(values)


def max_value(values):
    """Retourne la valeur maximale"""
    if not values:
        return float('nan')
    return max(values)


def percentile(values, p):
    """Calcule le percentile p (0-100)"""
    if not values:
        return float('nan')
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Méthode linéaire d'interpolation
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def range_value(values):
    """Calcule l'étendue (range) = max - min"""
    if not values:
        return float('nan')
    return max_value(values) - min_value(values)


def iqr(values):
    """Calcule l'écart interquartile (IQR) = Q3 - Q1"""
    if not values:
        return float('nan')
    return percentile(values, 75) - percentile(values, 25)


def skewness(values):
    """Calcule le coefficient d'asymétrie (skewness)"""
    if not values or len(values) < 3:
        return float('nan')
    
    n = len(values)
    mean_val = mean(values)
    std_val = std(values)
    
    if std_val == 0 or math.isnan(std_val):
        return float('nan')
    
    # Formule: E[((X - μ) / σ)³]
    sum_cubed = sum(((x - mean_val) / std_val) ** 3 for x in values)
    return sum_cubed / n


def kurtosis(values):
    """Calcule le coefficient d'aplatissement (kurtosis)"""
    if not values or len(values) < 4:
        return float('nan')
    
    n = len(values)
    mean_val = mean(values)
    std_val = std(values)
    
    if std_val == 0 or math.isnan(std_val):
        return float('nan')
    
    # Formule: E[((X - μ) / σ)⁴] - 3 (excess kurtosis)
    sum_fourth = sum(((x - mean_val) / std_val) ** 4 for x in values)
    return (sum_fourth / n) - 3


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
            'IQR': iqr(values),
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
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Range', 'IQR', 'Skewness', 'Kurtosis']
    columns = list(stats.keys())
    
    # Séparer les noms de colonnes en lignes
    col_lines = []
    max_lines = 1
    for col in columns:
        lines = col.split('\n')
        col_lines.append(lines)
        max_lines = max(max_lines, len(lines))
    
    # Calculer la largeur optimale pour chaque colonne
    col_widths = {}
    for i, col in enumerate(columns):
        # Largeur minimale = longueur max des lignes du nom
        max_width = max(len(line) for line in col_lines[i])
        
        # Vérifier la largeur nécessaire pour chaque statistique
        for stat_name in stat_names:
            value = stats[col][stat_name]
            if math.isnan(value):
                width = 3  # "NaN"
            elif stat_name == 'Count':
                width = len(f"{value:.0f}")
            else:
                width = len(f"{value:.6f}")
            max_width = max(max_width, width)
        
        # Ajouter 2 espaces de padding
        col_widths[col] = max_width + 2
    
    first_col_width = max(len(name) for name in stat_names) + 2
    
    # En-tête sur plusieurs lignes
    for line_idx in range(max_lines):
        header = ' ' * first_col_width
        for i, col in enumerate(columns):
            # Prendre la ligne correspondante ou une chaîne vide
            line_text = col_lines[i][line_idx] if line_idx < len(col_lines[i]) else ''
            header += f"{line_text:>{col_widths[col]}}"
        print(header)
    
    # Lignes de statistiques
    for stat_name in stat_names:
        row = f"{stat_name:<{first_col_width}}"
        for col in columns:
            value = stats[col][stat_name]
            width = col_widths[col]
            if math.isnan(value):
                row += f"{'NaN':>{width}}"
            elif stat_name == 'Count':
                row += f"{value:>{width}.0f}"
            else:
                row += f"{value:>{width}.6f}"
        print(row)


def main():
    """Fonction principale"""
    if len(sys.argv) != 2:
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

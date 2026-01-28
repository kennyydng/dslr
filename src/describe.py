#!/usr/bin/env python3
import sys

from math_utils import (
    ft_length,
    ft_min,
    ft_max,
    merge_sort,
    str_length,
    is_nan,
    ft_floor,
    ft_ceil,
    ft_mean,
    ft_std,
    parse_float,
    read_csv
)

def format_col_name(name):
    """Change le format des noms de colonnes trop longs"""
    split_names = {
        'Defense Against the Dark Arts': 'Defense\nagainst\n Dark Arts',
        'Care of Magical Creatures': 'Care of\nMagical\nCreatures',
        'History of Magic': 'Hist of\nMagic',
        'Transfiguration': 'Transfig',
        'Muggle Studies': 'Muggle\nStudies',
        'Ancient Runes': 'Ancient\nRunes'
    }
    return split_names.get(name, name)


def extract_numeric_data(headers, data):
    """Extrait les statistiques (donnees numerique)"""
    numeric_data = {}
    excluded_columns = {'Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Hogwarts House'}
    
    for header in headers:
        if header in excluded_columns:
            continue
        column_values = []
        
        for row in data:
            raw = row.get(header, '')
            value = parse_float(raw)
            if value is not None:
                column_values.append(value)
        
        if column_values:
            short_name = format_col_name(header)
            numeric_data[short_name] = column_values
    
    return numeric_data


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
    return ft_max(values) - ft_min(values)



def skewness(values):
    """Calcule le coefficient d'asymétrie (skewness)"""
    if ft_length(values) < 3:
        return float('nan')
    
    n = ft_length(values)
    mean_val = ft_mean(values)
    std_val = ft_std(values)
    
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
    mean_val = ft_mean(values)
    std_val = ft_std(values)
    
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
            'Count': ft_length(values),
            'Mean': ft_mean(values),
            'Std': ft_std(values),
            '25%': percentile(values, 25),
            '50%': percentile(values, 50),
            '75%': percentile(values, 75),
            'Min': ft_min(values),
            'Max': ft_max(values),
            'Range': range_value(values),
            'Skewness': skewness(values),
            'Kurtosis': kurtosis(values)
        }
    
    return stats


def display_statistics(stats):
    """Affiche les statistiques sous forme de tableau formaté"""
    if not stats:
        print("no data found.")
        return
    
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
        max_width = 0
        for line in col_lines[i]:
            w = str_length(line)
            if w > max_width:
                max_width = w
        
        # Vérifier la largeur nécessaire pour chaque stats
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
    
    for line_idx in range(max_lines):
        header = ' ' * first_col_width
        for i, col in enumerate(columns):
            # Prendre la ligne correspondante ou une chaîne vide
            line_text = col_lines[i][line_idx] if line_idx < ft_length(col_lines[i]) else ''
            header += f"{line_text:>{col_widths[col]}}"
        print(header)
    
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
    if ft_length(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>", file=sys.stderr)
        sys.exit(1)
    
    filepath = sys.argv[1]
    data = read_csv(filepath)
    
    # Extraire les headers depuis les clés du premier dictionnaire
    headers = list(data[0].keys()) if data else []
    
    numeric_data = extract_numeric_data(headers, data)
    stats = calculate_statistics(numeric_data)
    display_statistics(stats)


if __name__ == "__main__":
    main()

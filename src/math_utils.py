from __future__ import annotations
import sys
import csv


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


def ft_length(items) -> int:
    """Calcule la longueur d'un itérable."""
    n = 0
    for _ in items:
        n += 1
    return n


def str_length(text: str) -> int:
    """Calcule la longueur d'une chaîne de caractères."""
    n = 0
    for _ in text:
        n += 1
    return n


def ft_sum(values) -> float:
    """Calcule la somme des éléments d'un itérable."""
    total = 0.0
    for x in values:
        total += x
    return total


def ft_min(values) -> float:
    """Trouve la valeur minimale dans un itérable."""
    n = ft_length(values)
    if n == 0:
        return float('nan')
    m = values[0]
    for x in values:
        if x < m:
            m = x
    return m


def ft_max(values) -> float:
    """Trouve la valeur maximale dans un itérable."""
    n = ft_length(values)
    if n == 0:
        return float('nan')
    m = values[0]
    for x in values:
        if x > m:
            m = x
    return m


def clamp(value: float, low: float, high: float) -> float:
    """Restreint une valeur à un intervalle [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def is_nan(value: float) -> bool:
    """Vérifie si une valeur est NaN (Not a Number)."""
    # NaN is the only float that is not equal to itself.
    return value != value

def parse_float(value):
    """Convertit une valeur en float, retourne None si impossible."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def ft_floor(x: float) -> int:
    """Calcule la partie entière inférieure (plus grand entier <= x)."""
    if is_nan(x):
        return 0
    i = int(x)
    if x < 0 and x != i:
        return i - 1
    return i


def ft_ceil(x: float) -> int:
    """Calcule la partie entière supérieure (plus petit entier >= x)."""
    if is_nan(x):
        return 0
    i = int(x)
    if x > 0 and x != i:
        return i + 1
    return i


def ft_abs(x: float) -> float:
    """Calcule la valeur absolue d'un nombre."""
    if x < 0:
        return -x
    return x


def ft_sqrt(x: float) -> float:
    """Calcule la racine carrée via la méthode de Newton-Raphson."""
    if is_nan(x) or x < 0:
        return float('nan')
    if x == 0.0:
        return 0.0
    guess = x if x >= 1 else 1.0
    for _ in range(64):
        next_guess = 0.5 * (guess + x / guess)
        if ft_abs(next_guess - guess) < 1e-15:
            break
        guess = next_guess
    return guess


def ft_exp(x: float) -> float:
    """Calcule l'exponentielle via la série de Taylor."""
    if is_nan(x):
        return float('nan')
    # Handle large negative/positive to avoid overflow
    if x > 709:
        return float('inf')
    if x < -709:
        return 0.0
    # exp(x) = sum_{n=0}^{inf} x^n / n!
    result = 1.0
    term = 1.0
    for n in range(1, 300):
        term *= x / n
        result += term
        if ft_abs(term) < 1e-15:
            break
    return result


def ft_log(x: float) -> float:
    """Calcule le logarithme naturel via Newton-Raphson sur exp."""
    if is_nan(x) or x <= 0:
        return float('nan')
    # Initial guess
    guess = 0.0
    if x > 1:
        guess = x / 2.7
    else:
        guess = x - 1
    for _ in range(100):
        e = ft_exp(guess)
        if e == 0:
            break
        next_guess = guess + (x - e) / e
        if ft_abs(next_guess - guess) < 1e-15:
            break
        guess = next_guess
    return guess


def ft_mean(values) -> float:
    """Calcule la moyenne arithmétique."""
    n = ft_length(values)
    if n == 0:
        return float('nan')
    return ft_sum(values) / n


def ft_variance(values, mean_val: float | None = None) -> float:
    """Calcule la variance."""
    n = ft_length(values)
    if n == 0:
        return float('nan')
    if mean_val is None:
        mean_val = ft_mean(values)
    acc = 0.0
    count = 0
    for x in values:
        diff = x - mean_val
        acc += diff * diff
        count += 1
    if count == 0:
        return float('nan')
    return acc / count


def ft_std(values) -> float:
    """Calcule l'écart-type."""
    v = ft_variance(values)
    if is_nan(v):
        return float('nan')
    return ft_sqrt(v)


def merge_sorted(left, right):
    """Fusionne deux listes triées en une seule liste triée."""
    merged = []
    i = 0
    j = 0
    left_n = ft_length(left)
    right_n = ft_length(right)
    while i < left_n and j < right_n:
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    while i < left_n:
        merged.append(left[i])
        i += 1
    while j < right_n:
        merged.append(right[j])
        j += 1
    return merged


def merge_sort(values):
    """Trie une liste en utilisant l'algorithme de tri fusion (mergesort)."""
    n = ft_length(values)
    if n <= 1:
        return values[:]
    mid = n // 2
    left = merge_sort(values[:mid])
    right = merge_sort(values[mid:])
    return merge_sorted(left, right)


def sort_pairs_by_value(pairs, reverse: bool = False):
    """Trie des paires (clé, valeur) par valeur en utilisant un tri fusion stable."""   
    n = ft_length(pairs)
    if n <= 1:
        return pairs[:]

    mid = n // 2
    left = sort_pairs_by_value(pairs[:mid], reverse=reverse)
    right = sort_pairs_by_value(pairs[mid:], reverse=reverse)

    merged = []
    i = 0
    j = 0
    left_n = ft_length(left)
    right_n = ft_length(right)

    def before(a, b):
        if reverse:
            return a[1] >= b[1]
        return a[1] <= b[1]

    while i < left_n and j < right_n:
        if before(left[i], right[j]):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    while i < left_n:
        merged.append(left[i])
        i += 1
    while j < right_n:
        merged.append(right[j])
        j += 1

    return merged


def argmax_dict(dct):
    """Retourne la clé correspondant à la valeur maximale dans un dictionnaire."""
    best_key = None
    best_val = None
    for k, v in dct.items():
        if best_key is None or v > best_val:
            best_key = k
            best_val = v
    return best_key


def argmin_dict(dct):
    """Retourne la clé correspondant à la valeur minimale dans un dictionnaire."""
    best_key = None
    best_val = None
    for k, v in dct.items():
        if best_key is None or v < best_val:
            best_key = k
            best_val = v
    return best_key


def count_occurrences(items):
    """Compte les occurrences de chaque élément dans un itérable."""
    counts = {}
    for x in items:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

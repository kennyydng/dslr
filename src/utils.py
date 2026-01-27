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


def pearson_corr(x_values, y_values) -> float:
    """Calcule la corrélation de Pearson entre deux variables (ignore les valeurs None).
    
    Formule: r = Σ(xi - x̄)(yi - ȳ) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
    
    Retourne une valeur entre -1 et +1:
    - r = +1: corrélation positive parfaite (relation linéaire croissante)
    - r = 0: pas de corrélation linéaire
    - r = -1: corrélation négative parfaite (relation linéaire décroissante)
    """
    # ÉTAPE 1: Nettoyer les données en retirant les paires avec valeurs manquantes (None)
    xs = []
    ys = []
    i = 0
    n = ft_length(x_values)
    while i < n:
        x = x_values[i]
        y = y_values[i]
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
        i += 1

    # ÉTAPE 2: Vérifier qu'on a suffisamment de données
    m = ft_length(xs)
    if m < 2:  # Besoin d'au moins 2 points pour calculer une corrélation
        return 0.0

    # ÉTAPE 3: Calculer les moyennes de X et Y
    mx = ft_mean(xs)  
    my = ft_mean(ys) 

    # ÉTAPE 4: Calculer les composantes de la formule de Pearson
    num = 0.0   
    den_x = 0.0 
    den_y = 0.0
    
    i = 0
    while i < m:
        dx = xs[i] - mx    
        dy = ys[i] - my    
        
        num += dx * dy     
        den_x += dx * dx   
        den_y += dy * dy
        i += 1

    # ÉTAPE 5: Vérifier la division par zéro
    if den_x == 0.0 or den_y == 0.0:
        return 0.0  # Pas de corrélation définie si l'une des variables ne varie pas

    # ÉTAPE 6: Calculer le coefficient de corrélation de Pearson
    # Division par sqrt(den_x * den_y) pour normaliser dans [-1, +1]
    return num / ft_sqrt(den_x * den_y)


def normalize_features(X):
    """Normalise les features avec standardisation (Z-score normalization).
    
    Formule: x_normalized = (x - mean) / std
    
    Cette normalisation est essentielle pour le gradient descent car:
    - Évite que les features avec grandes valeurs dominent le gradient
    - Accélère la convergence (toutes les features sur la même échelle)
    - Rend le learning rate plus facile à régler
    
    Args:
        X: Liste de listes (chaque sous-liste = features d'un exemple)
        
    Returns:
        (X_normalized, means, stds): Données normalisées + paramètres de normalisation
    """
    n_features = ft_length(X[0])
    means = []
    stds = []
    
    # ÉTAPE 1: Calculer moyenne et écart-type pour chaque feature (colonne)
    for j in range(n_features):
        # Extraire toutes les valeurs de la feature j
        values = []
        for row in X:
            values.append(row[j])

        # Calculer statistiques
        m = ft_mean(values)
        s = ft_std(values)
        
        # Éviter division par zéro si la feature est constante
        if is_nan(s) or s == 0.0:
            s = 1.0

        means.append(m)
        stds.append(s)
    
    # ÉTAPE 2: Normaliser chaque valeur avec (x - mean) / std
    X_normalized = []
    for row in X:
        normalized_row = []
        j = 0
        while j < n_features:
            # Appliquer la normalisation Z-score
            normalized_value = (row[j] - means[j]) / stds[j]
            normalized_row.append(normalized_value)
            j += 1
        X_normalized.append(normalized_row)
    
    return X_normalized, means, stds


def sigmoid(z):
    """Fonction sigmoïde : convertit un score linéaire en probabilité [0, 1].
    
    Formule: σ(z) = 1 / (1 + e^(-z))
    
    Propriétés:
    - σ(0) = 0.5
    - σ(+∞) → 1
    - σ(-∞) → 0
    
    Le clipping évite les overflow numériques lors du calcul de e^(-z).
    """
    z = clamp(z, -500, 500)  # Éviter overflow avec e^(très grand nombre)
    return 1.0 / (1.0 + ft_exp(-z))


def predict_probability(X, weights):
    """Calcule la probabilité avec la régression logistique.
    
    Formule: P(y=1|X) = σ(w0 + w1*x1 + w2*x2 + ... + wn*xn)
    
    Args:
        X: Features d'un exemple (liste de valeurs)
        weights: Poids du modèle [biais, w1, w2, ..., wn]
        
    Returns:
        Probabilité entre 0 et 1
    """
    z = weights[0]  # Biais (intercept)
    i = 0
    n = ft_length(X)
    while i < n:
        z += weights[i + 1] * X[i]  # Somme pondérée
        i += 1
    return sigmoid(z)


def compute_cost(X, y_binary, weights):
    """Calcule le coût (log loss) pour la régression logistique.
    
    Formule: L = -1/m * Σ[y*log(h) + (1-y)*log(1-h)]
    
    Plus le coût est faible, meilleur est le modèle.
    
    Args:
        X: Données d'entraînement (liste de listes)
        y_binary: Labels binaires (0 ou 1)
        weights: Poids actuels du modèle
        
    Returns:
        Coût moyen sur tous les exemples
    """
    m = ft_length(X)
    total_cost = 0.0
    
    for i in range(m):
        h = predict_probability(X[i], weights)
        # Éviter log(0) qui donnerait -inf
        h = clamp(h, 1e-15, 1 - 1e-15)
        # Log-loss pour un exemple
        cost = -y_binary[i] * ft_log(h) - (1 - y_binary[i]) * ft_log(1 - h)
        total_cost += cost
    
    return total_cost / m


def extract_features_and_labels(data, selected_features):
    """Extrait les features et labels du dataset.
    
    Filtre les lignes avec valeurs manquantes et ne garde que les maisons valides.
    
    Args:
        data: Dataset (liste de dictionnaires)
        selected_features: Liste des noms de features à extraire
        
    Returns:
        (X, y): Features normalisées et labels
    """
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    
    X = []
    y = []
    
    for row in data:
        house = row.get('Hogwarts House')
        if house not in houses:
            continue
        
        # Extraire les features et vérifier qu'aucune n'est None
        features = []
        valid = True
        for feature in selected_features:
            value = parse_float(row.get(feature))
            if value is None:
                valid = False
                break
            features.append(value)
        
        # Ne garder que les exemples complets
        if valid:
            X.append(features)
            y.append(house)
    
    return X, y


def save_weights(models, means, stds, selected_features, filepath='weights.json'):
    """Sauvegarde les poids et paramètres de normalisation dans un fichier JSON.
    
    Args:
        models: Dictionnaire {house: weights}
        means: Moyennes de normalisation pour chaque feature
        stds: Écarts-types de normalisation pour chaque feature
        selected_features: Liste des noms de features utilisées
        filepath: Chemin du fichier de sortie
    """
    import json
    
    data = {
        'features': selected_features,
        'normalization': {
            'means': means,
            'stds': stds
        },
        'models': {house: weights for house, weights in models.items()}
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Poids sauvegardés dans '{filepath}'")

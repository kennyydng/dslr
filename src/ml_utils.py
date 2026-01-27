from math_utils import (
    ft_length, ft_mean, ft_std, ft_sqrt, ft_exp, ft_log,
    is_nan, clamp, parse_float
)


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


def normalize_features_with_params(features, means, stds):
    """Normalise des features avec des paramètres de normalisation existants.
    
    Version pour la prédiction : utilise les moyennes et écarts-types
    calculés lors de l'entraînement.
    
    Args:
        features: Liste de valeurs à normaliser
        means: Moyennes calculées sur le training set
        stds: Écarts-types calculés sur le training set
        
    Returns:
        Liste de valeurs normalisées
    """
    normalized = []
    i = 0
    n = ft_length(features)
    while i < n:
        normalized.append((features[i] - means[i]) / stds[i])
        i += 1
    return normalized


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

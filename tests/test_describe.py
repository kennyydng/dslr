#!/usr/bin/env python3
"""
Tests unitaires pour les fonctions de describe.py
"""

import sys
import math
sys.path.insert(0, '.')
from describe import (
    count, mean, std, min_value, max_value, 
    percentile, variance, parse_float
)


def test_parse_float():
    """Test de la fonction parse_float"""
    assert parse_float("42.5") == 42.5
    assert parse_float("-10") == -10.0
    assert parse_float("abc") is None
    assert parse_float("") is None
    assert parse_float(None) is None
    print("✓ test_parse_float passed")


def test_count():
    """Test de la fonction count"""
    assert count([1, 2, 3, 4, 5]) == 5
    assert count([]) == 0
    assert count([1]) == 1
    print("✓ test_count passed")


def test_mean():
    """Test de la fonction mean"""
    assert mean([1, 2, 3, 4, 5]) == 3.0
    assert mean([10]) == 10.0
    assert math.isnan(mean([]))
    assert mean([0, 0, 0]) == 0.0
    print("✓ test_mean passed")


def test_variance():
    """Test de la fonction variance"""
    values = [1, 2, 3, 4, 5]
    mean_val = mean(values)
    var = variance(values, mean_val)
    assert abs(var - 2.0) < 0.001  # Variance théorique = 2.0
    print("✓ test_variance passed")


def test_std():
    """Test de la fonction std"""
    values = [1, 2, 3, 4, 5]
    std_val = std(values)
    expected_std = math.sqrt(2.0)  # sqrt(variance)
    assert abs(std_val - expected_std) < 0.001
    print("✓ test_std passed")


def test_min_max():
    """Test des fonctions min et max"""
    values = [1, 5, 3, 9, 2]
    assert min_value(values) == 1
    assert max_value(values) == 9
    assert min_value([42]) == 42
    assert max_value([42]) == 42
    assert math.isnan(min_value([]))
    assert math.isnan(max_value([]))
    print("✓ test_min_max passed")


def test_percentile():
    """Test de la fonction percentile"""
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Test médiane (50%)
    assert percentile(values, 50) == 5.5
    
    # Test premier quartile (25%)
    q1 = percentile(values, 25)
    assert abs(q1 - 3.25) < 0.001
    
    # Test troisième quartile (75%)
    q3 = percentile(values, 75)
    assert abs(q3 - 7.75) < 0.001
    
    # Test min et max
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 10
    
    print("✓ test_percentile passed")


def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*50)
    print("Exécution des tests unitaires")
    print("="*50 + "\n")
    
    test_parse_float()
    test_count()
    test_mean()
    test_variance()
    test_std()
    test_min_max()
    test_percentile()
    
    print("\n" + "="*50)
    print("✓ Tous les tests sont passés avec succès !")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

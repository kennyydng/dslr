"""Shared manual helpers for DSLR.

The project constraints forbid using high-level helpers that compute statistics for you
(e.g. mean/std/min/max/percentile/corrcoef/describe). This module provides loop-based
implementations that can be reused across scripts.
"""

from __future__ import annotations


def ft_length(items) -> int:
    n = 0
    for _ in items:
        n += 1
    return n


def str_length(text: str) -> int:
    n = 0
    for _ in text:
        n += 1
    return n


def ft_sum(values) -> float:
    total = 0.0
    for x in values:
        total += x
    return total


def ft_min(values) -> float:
    n = ft_length(values)
    if n == 0:
        return float('nan')
    m = values[0]
    for x in values:
        if x < m:
            m = x
    return m


def ft_max(values) -> float:
    n = ft_length(values)
    if n == 0:
        return float('nan')
    m = values[0]
    for x in values:
        if x > m:
            m = x
    return m


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def is_nan(value: float) -> bool:
    # NaN is the only float that is not equal to itself.
    return value != value


def ft_floor(x: float) -> int:
    """Manual floor (largest integer <= x)."""
    if is_nan(x):
        return 0
    i = int(x)
    if x < 0 and x != i:
        return i - 1
    return i


def ft_ceil(x: float) -> int:
    """Manual ceil (smallest integer >= x)."""
    if is_nan(x):
        return 0
    i = int(x)
    if x > 0 and x != i:
        return i + 1
    return i


def ft_abs(x: float) -> float:
    if x < 0:
        return -x
    return x


def ft_sqrt(x: float) -> float:
    """Manual square root using Newton-Raphson."""
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
    """Manual exp using Taylor series."""
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
    """Manual natural log using Newton-Raphson on exp."""
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
    n = ft_length(values)
    if n == 0:
        return float('nan')
    return ft_sum(values) / n


def ft_variance(values, mean_val: float | None = None) -> float:
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
    v = ft_variance(values)
    if is_nan(v):
        return float('nan')
    return ft_sqrt(v)


def merge_sorted(left, right):
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
    n = ft_length(values)
    if n <= 1:
        return values[:]
    mid = n // 2
    left = merge_sort(values[:mid])
    right = merge_sort(values[mid:])
    return merge_sorted(left, right)


def sort_pairs_by_value(pairs, reverse: bool = False):
    """Stable mergesort for list of (key, value) pairs by value."""

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
    best_key = None
    best_val = None
    for k, v in dct.items():
        if best_key is None or v > best_val:
            best_key = k
            best_val = v
    return best_key


def argmin_dict(dct):
    best_key = None
    best_val = None
    for k, v in dct.items():
        if best_key is None or v < best_val:
            best_key = k
            best_val = v
    return best_key


def count_occurrences(items):
    counts = {}
    for x in items:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts


def pearson_corr(x_values, y_values) -> float:
    """Pearson correlation (filters out missing pairs where value is None)."""

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

    m = ft_length(xs)
    if m < 2:
        return 0.0

    mx = ft_mean(xs)
    my = ft_mean(ys)

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

    if den_x == 0.0 or den_y == 0.0:
        return 0.0

    return num / ft_sqrt(den_x * den_y)

"""Pure NumPy functions for Adstock decay and Hill saturation transformations.

These are the mathematical building blocks of the Marketing Mix Model:
- Adstock: models the lagged/carryover effect of media spend over time
- Hill: models the diminishing returns (saturation) of media investment
"""

import numpy as np
from typing import Tuple


def apply_adstock(x: np.ndarray, decay_rate: float) -> np.ndarray:
    """Apply geometric Adstock transformation to model media carryover effects.

    The Adstock transformation captures how advertising impact persists beyond
    the week it was aired. Each period inherits a fraction (decay_rate) of the
    previous period's transformed value.

    Formula: adstock[t] = x[t] + decay_rate * adstock[t-1]

    Args:
        x: Raw spend series (1D array of non-negative values).
        decay_rate: Retention rate per period (0.0 = no memory, 1.0 = full memory).

    Returns:
        Transformed series with carryover effects applied.
    """
    if not 0.0 <= decay_rate <= 1.0:
        raise ValueError(f"decay_rate must be in [0, 1], got {decay_rate}")

    x = np.asarray(x, dtype=np.float64)
    adstocked = np.zeros_like(x)
    adstocked[0] = x[0]

    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay_rate * adstocked[t - 1]

    return adstocked


def compute_adstock_halflife(decay_rate: float) -> float:
    """Compute the half-life (in periods) of the Adstock decay.

    The half-life indicates how many periods it takes for the carryover
    effect to decay to 50% of its initial value.

    Formula: halflife = log(0.5) / log(decay_rate)

    Args:
        decay_rate: Retention rate per period.

    Returns:
        Half-life in number of periods. Returns 0.0 if decay_rate is 0.
    """
    if decay_rate <= 0.0:
        return 0.0
    if decay_rate >= 1.0:
        return float("inf")
    return np.log(0.5) / np.log(decay_rate)


def apply_hill_saturation(x: np.ndarray, K: float, S: float) -> np.ndarray:
    """Apply Hill function saturation curve to model diminishing returns.

    The Hill function is a sigmoidal curve that maps spend to a 0-1 scale,
    capturing the economic reality that each additional dollar spent yields
    progressively less incremental return.

    Formula: hill(x) = x^S / (K^S + x^S)

    Where:
        K = Half-saturation point (spend level producing 50% of max effect)
        S = Shape/steepness parameter (higher = sharper transition)

    Args:
        x: Input series (typically Adstock-transformed spend).
        K: Half-saturation constant (must be > 0).
        S: Shape parameter controlling steepness (must be > 0).

    Returns:
        Saturated series with values in [0, 1].
    """
    if K <= 0:
        raise ValueError(f"K (half-saturation) must be > 0, got {K}")
    if S <= 0:
        raise ValueError(f"S (shape) must be > 0, got {S}")

    x = np.asarray(x, dtype=np.float64)
    x_safe = np.maximum(x, 0.0)
    return np.power(x_safe, S) / (np.power(K, S) + np.power(x_safe, S))


def compute_marginal_roas(spend_level: float, K: float, S: float, beta: float) -> float:
    """Compute the marginal ROAS at a given spend level.

    The marginal ROAS is the derivative of the Hill-transformed revenue
    with respect to spend, indicating how much additional revenue one
    extra dollar of spend would generate at the current level.

    Formula: d/dx [beta * x^S / (K^S + x^S)]
           = beta * S * K^S * x^(S-1) / (K^S + x^S)^2

    Args:
        spend_level: Current spend level to evaluate.
        K: Half-saturation constant.
        S: Shape parameter.
        beta: Model coefficient (revenue multiplier for this channel).

    Returns:
        Marginal ROAS value (additional revenue per additional BRL spent).
    """
    if spend_level <= 0:
        return beta * S / K if K > 0 else 0.0

    numerator = beta * S * np.power(K, S) * np.power(spend_level, S - 1)
    denominator = np.power(np.power(K, S) + np.power(spend_level, S), 2)
    return float(numerator / denominator)


def classify_channel_health(marginal_roas: float) -> str:
    """Classify channel investment efficiency based on marginal ROAS.

    Args:
        marginal_roas: Current marginal return on ad spend.

    Returns:
        Health classification string.
    """
    if marginal_roas >= 3.0:
        return "SCALING"
    elif marginal_roas >= 1.5:
        return "EFFICIENT"
    elif marginal_roas >= 0.8:
        return "SATURATED"
    else:
        return "OVER_INVESTED"

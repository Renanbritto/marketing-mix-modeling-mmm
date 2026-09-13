"""Unit tests for pure econometric transformations (Adstock and Hill)."""

import pytest
import numpy as np

from src.engine.transformations import (
    apply_adstock,
    compute_adstock_halflife,
    apply_hill_saturation,
    compute_marginal_roas,
    classify_channel_health,
)


class TestAdstockTransformation:
    """Test suite for Adstock temporal carryover."""

    def test_adstock_zero_decay(self):
        """Decay rate of 0.0 should return identical input series (no carryover)."""
        x = np.array([100.0, 200.0, 150.0, 0.0])
        result = apply_adstock(x, decay_rate=0.0)
        np.testing.assert_array_almost_equal(result, x)

    def test_adstock_full_memory(self):
        """Decay rate of 1.0 should accumulate all past investments."""
        x = np.array([10.0, 20.0, 30.0])
        result = apply_adstock(x, decay_rate=1.0)
        expected = np.array([10.0, 30.0, 60.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_adstock_realistic_decay(self):
        """Decay rate of 0.5 should retain 50% from prior period."""
        x = np.array([100.0, 0.0, 0.0])
        result = apply_adstock(x, decay_rate=0.5)
        expected = np.array([100.0, 50.0, 25.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_adstock_invalid_decay_bounds(self):
        """Decay rate outside [0, 1] must raise ValueError."""
        with pytest.raises(ValueError):
            apply_adstock(np.array([10.0]), decay_rate=-0.1)
        with pytest.raises(ValueError):
            apply_adstock(np.array([10.0]), decay_rate=1.1)

    def test_adstock_halflife_calculation(self):
        """Check half-life calculation accuracy."""
        assert compute_adstock_halflife(0.0) == 0.0
        assert compute_adstock_halflife(0.5) == pytest.approx(1.0, 0.01)
        assert compute_adstock_halflife(1.0) == float("inf")


class TestHillSaturation:
    """Test suite for Hill diminishing returns saturation curves."""

    def test_hill_output_bounds(self):
        """Hill output must be strictly bounded between 0.0 and 1.0."""
        x = np.array([0.0, 1000.0, 50000.0, 1000000.0])
        result = apply_hill_saturation(x, K=50000.0, S=2.0)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_hill_half_saturation_point(self):
        """At spend x = K, Hill saturation must equal exactly 0.5."""
        K = 40000.0
        result = apply_hill_saturation(np.array([K]), K=K, S=2.0)
        assert result[0] == pytest.approx(0.5, 1e-4)

    def test_hill_zero_spend(self):
        """Zero spend should produce zero effect."""
        result = apply_hill_saturation(np.array([0.0]), K=10000.0, S=2.0)
        assert result[0] == 0.0

    def test_hill_invalid_parameters(self):
        """K <= 0 or S <= 0 must raise ValueError."""
        with pytest.raises(ValueError):
            apply_hill_saturation(np.array([100.0]), K=-10.0, S=1.5)
        with pytest.raises(ValueError):
            apply_hill_saturation(np.array([100.0]), K=10.0, S=0.0)

    def test_marginal_roas_diminishing(self):
        """Marginal ROAS must decline once spend is past the inflection point."""
        K = 20000.0
        S = 2.0
        beta = 100000.0
        
        m1 = compute_marginal_roas(25000.0, K, S, beta)
        m2 = compute_marginal_roas(50000.0, K, S, beta)
        m3 = compute_marginal_roas(100000.0, K, S, beta)
        
        assert m1 > m2 > m3

    def test_channel_health_classification(self):
        """Verify health classifications based on mROAS thresholds."""
        assert classify_channel_health(3.5) == "SCALING"
        assert classify_channel_health(2.0) == "EFFICIENT"
        assert classify_channel_health(1.0) == "SATURATED"
        assert classify_channel_health(0.5) == "OVER_INVESTED"

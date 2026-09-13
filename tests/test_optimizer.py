"""Unit tests for the BudgetOptimizer (SciPy SLSQP)."""

import pytest
import numpy as np

from src.engine.mmm_model import MarketingMixModel
from src.engine.optimizer import BudgetOptimizer
from src.data.sample_generator import MarketingDataGenerator


@pytest.fixture
def fitted_model():
    """Provide a fitted MMM model for optimization tests."""
    gen = MarketingDataGenerator(seed=123)
    df = gen.generate(n_weeks=52)
    model = MarketingMixModel().fit(df)
    return model


class TestBudgetOptimizer:
    """Test suite for SciPy-based budget optimization."""

    def test_budget_equality_constraint(self, fitted_model):
        """Sum of optimal channel allocations must equal total requested budget."""
        optimizer = BudgetOptimizer(fitted_model)
        target_budget = 120_000.0

        result = optimizer.optimize(total_budget=target_budget)
        allocated_sum = sum(result["optimal_allocation"].values())

        assert allocated_sum == pytest.approx(target_budget, abs=1.0)

    def test_channel_bounds_constraints(self, fitted_model):
        """Per-channel min and max boundaries must be strictly respected."""
        optimizer = BudgetOptimizer(fitted_model)
        target_budget = 100_000.0
        min_crm = 5_000.0
        max_meta = 40_000.0

        result = optimizer.optimize(
            total_budget=target_budget,
            min_per_channel={"crm_email": min_crm},
            max_per_channel={"meta_ads": max_meta},
        )

        alloc = result["optimal_allocation"]
        assert alloc["crm_email"] >= min_crm - 0.1
        assert alloc["meta_ads"] <= max_meta + 0.1

    def test_optimal_yields_positive_or_zero_lift(self, fitted_model):
        """Optimal allocation must yield >= revenue than arbitrary uniform distribution."""
        optimizer = BudgetOptimizer(fitted_model)
        result = optimizer.optimize(total_budget=80_000.0)

        assert result["lift_vs_uniform_pct"] >= -0.01

    def test_result_structure_completeness(self, fitted_model):
        """Check all required analytical output keys."""
        optimizer = BudgetOptimizer(fitted_model)
        result = optimizer.optimize(total_budget=90_000.0)

        assert "optimal_allocation" in result
        assert "projected_revenue" in result
        assert "roas_global" in result
        assert "channels" in result
        assert len(result["channels"]) == 5

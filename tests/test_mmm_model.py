"""Unit tests for the core MarketingMixModel econometric pipeline."""

import pytest
import numpy as np
import pandas as pd

from src.engine.mmm_model import MarketingMixModel
from src.data.sample_generator import MarketingDataGenerator
from src.config.settings import settings


@pytest.fixture
def sample_dataset():
    """Provide reproducible 52-week marketing dataset."""
    gen = MarketingDataGenerator(seed=42)
    return gen.generate(n_weeks=52)


class TestMarketingMixModel:
    """Test suite for MarketingMixModel."""

    def test_model_initialization(self):
        """Model initializes in unfitted state with proper channels."""
        model = MarketingMixModel(alpha=1.0)
        assert not model._fitted
        assert len(model.channels) == 5

    def test_model_fit_and_metrics(self, sample_dataset):
        """Model fits and achieves high R-squared on ground-truth data."""
        model = MarketingMixModel(alpha=1.0)
        model.fit(sample_dataset)

        assert model._fitted
        metrics = model.get_metrics()
        assert metrics["r_squared"] > 0.80
        assert metrics["mape_pct"] < 20.0
        assert metrics["n_observations"] == 52

    def test_model_predict_shape(self, sample_dataset):
        """Predictions must match length of test dataset."""
        model = MarketingMixModel().fit(sample_dataset)
        preds = model.predict(sample_dataset)
        assert len(preds) == len(sample_dataset)
        assert np.all(preds > 0)

    def test_model_decomposition_completeness(self, sample_dataset):
        """Decomposition must account for all channels, base organic, and match total revenue."""
        model = MarketingMixModel().fit(sample_dataset)
        decomp = model.decompose()

        assert "total_actual_revenue" in decomp
        assert "base_organic_total" in decomp
        assert len(decomp["channel_contributions"]) == 5
        assert len(decomp["weekly_data"]) == 52

        # Sum of weekly predicted should match total_predicted_revenue
        pred_sum = sum(w["total_predicted"] for w in decomp["weekly_data"])
        assert pred_sum == pytest.approx(decomp["total_predicted_revenue"], 1.0)

    def test_model_simulation_whatif(self, sample_dataset):
        """What-if scenario simulation should return positive projected revenue."""
        model = MarketingMixModel().fit(sample_dataset)
        alloc = {
            "meta_ads": 30000.0,
            "google_ads": 25000.0,
            "tiktok_ads": 15000.0,
            "influencers": 10000.0,
            "crm_email": 5000.0,
        }
        res = model.simulate_scenario(alloc)
        assert res["total_weekly_budget"] == 85000.0
        assert res["projected_weekly_revenue"] > 100000.0
        assert res["roas_global"] > 1.0

    def test_model_response_curves(self, sample_dataset):
        """Response curves must contain points for every channel."""
        model = MarketingMixModel().fit(sample_dataset)
        curves = model.get_response_curves(n_points=20)
        assert len(curves) == 5
        for c in curves:
            assert len(c["points"]) == 20
            assert c["saturation_point"] > 0

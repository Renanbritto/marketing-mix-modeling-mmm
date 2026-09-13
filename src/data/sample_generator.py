"""Realistic marketing investment and revenue data generator.

Generates synthetic time series with:
- Channel-specific spend patterns with variance
- Seasonality (Black Friday, Christmas, January dip)
- Revenue driven by true Adstock + Hill mechanics + noise
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.engine.transformations import apply_adstock, apply_hill_saturation
from src.config.settings import settings


class MarketingDataGenerator:
    """Generate realistic weekly marketing spend and revenue data."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or settings.random_seed
        self.rng = np.random.default_rng(self.seed)

    def generate(self, n_weeks: Optional[int] = None) -> pd.DataFrame:
        """Generate a DataFrame with weekly spend per channel and resulting revenue.

        The revenue is generated using the true Adstock + Hill mechanics plus
        base organic revenue, seasonality, and stochastic noise. This ensures
        the MMM model can recover meaningful coefficients.

        Args:
            n_weeks: Number of weeks to generate (default: settings.n_weeks).

        Returns:
            DataFrame with columns: week, spend_<channel>, revenue.
        """
        n = n_weeks or settings.n_weeks
        weeks = np.arange(1, n + 1)

        data = {"week": weeks}

        # Generate spend per channel with realistic patterns
        spends = {}
        for ch, params in settings.channels.items():
            base_spend = params.weekly_budget_default
            # Weekly variance (+-25%)
            noise = self.rng.normal(1.0, 0.25, size=n)
            noise = np.clip(noise, 0.3, 2.0)

            spend = base_spend * noise

            # Seasonal uplift: Black Friday (week ~47), Christmas (week ~51)
            for w in range(n):
                week_of_year = (w % 52) + 1
                if week_of_year in [46, 47, 48]:  # Black Friday period
                    spend[w] *= 1.8
                elif week_of_year in [50, 51, 52]:  # Christmas
                    spend[w] *= 1.5
                elif week_of_year in [1, 2, 3, 4]:  # January dip
                    spend[w] *= 0.6

            # Slight upward trend over time
            trend = 1.0 + 0.001 * np.arange(n)
            spend *= trend

            spend = np.maximum(spend, 0.0)
            spends[ch] = spend
            data[f"spend_{ch}"] = np.round(spend, 2)

        # Generate revenue using true DGP (Data Generating Process)
        # True coefficients (what the model should approximately recover)
        true_betas = {
            "meta_ads": 320_000.0,
            "google_ads": 280_000.0,
            "tiktok_ads": 180_000.0,
            "influencers": 150_000.0,
            "crm_email": 95_000.0,
        }

        revenue = np.full(n, settings.base_revenue_weekly)

        # Add channel contributions via Adstock + Hill
        for ch, params in settings.channels.items():
            adstocked = apply_adstock(spends[ch], params.decay_rate)
            saturated = apply_hill_saturation(adstocked, params.half_saturation_K, params.shape_S)
            revenue += true_betas.get(ch, 100_000.0) * saturated

        # Add seasonality
        seasonality = 15_000.0 * np.sin(2 * np.pi * weeks / 52.0)
        revenue += seasonality

        # Add stochastic noise (5% relative)
        noise = self.rng.normal(0, 0.05, size=n)
        revenue *= (1.0 + noise)

        # Ensure positive revenue
        revenue = np.maximum(revenue, 50_000.0)

        data["revenue"] = np.round(revenue, 2)

        return pd.DataFrame(data)

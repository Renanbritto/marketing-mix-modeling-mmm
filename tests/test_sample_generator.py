"""Unit tests for synthetic marketing data generator."""

import pytest
import pandas as pd
import numpy as np

from src.data.sample_generator import MarketingDataGenerator
from src.config.settings import settings


class TestSampleGenerator:
    """Test suite for MarketingDataGenerator."""

    def test_generate_default_weeks(self):
        """Should generate exact number of default weeks with expected columns."""
        gen = MarketingDataGenerator(seed=42)
        df = gen.generate()

        assert len(df) == settings.n_weeks
        assert "week" in df.columns
        assert "revenue" in df.columns

        for ch in settings.channels:
            assert f"spend_{ch}" in df.columns

    def test_generate_non_negative_values(self):
        """All spends and revenue values must be strictly non-negative."""
        gen = MarketingDataGenerator(seed=99)
        df = gen.generate(n_weeks=26)

        for col in df.columns:
            assert (df[col] >= 0).all()

    def test_reproducibility(self):
        """Same seed must produce identical dataframes."""
        df1 = MarketingDataGenerator(seed=123).generate(n_weeks=20)
        df2 = MarketingDataGenerator(seed=123).generate(n_weeks=20)
        pd.testing.assert_frame_equal(df1, df2)

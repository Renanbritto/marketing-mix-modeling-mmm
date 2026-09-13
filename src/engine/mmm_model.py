"""Marketing Mix Model: fits Adstock + Hill + Ridge Regression pipeline.

This module implements the core econometric engine that:
1. Applies Adstock transformation to model media carryover
2. Applies Hill saturation to model diminishing returns
3. Fits a Ridge Regression to estimate channel coefficients
4. Decomposes total revenue into base + channel contributions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

from src.engine.transformations import (
    apply_adstock,
    apply_hill_saturation,
    compute_marginal_roas,
    classify_channel_health,
)
from src.config.settings import settings


class MarketingMixModel:
    """End-to-end Marketing Mix Model with Adstock + Hill + Ridge pipeline."""

    def __init__(self, alpha: float = 1.0):
        """Initialize the MMM engine.

        Args:
            alpha: Ridge regression regularization strength.
        """
        self.alpha = alpha
        self.model: Optional[Ridge] = None
        self.channels: List[str] = list(settings.channels.keys())
        self.coefficients: Dict[str, float] = {}
        self.intercept: float = 0.0
        self._fitted = False
        self._train_df: Optional[pd.DataFrame] = None
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._X_transformed: Optional[np.ndarray] = None
        self._seasonality_coef: float = 0.0

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw spend columns through Adstock -> Hill pipeline.

        Args:
            df: DataFrame with spend columns named as channel keys.

        Returns:
            Feature matrix (n_weeks x n_channels + 1) with Hill-saturated values + seasonality.
        """
        features = []

        for ch in self.channels:
            col = f"spend_{ch}"
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

            raw_spend = df[col].values
            params = settings.channels[ch]

            # Step 1: Adstock (temporal carryover)
            adstocked = apply_adstock(raw_spend, params.decay_rate)

            # Step 2: Hill saturation (diminishing returns)
            saturated = apply_hill_saturation(adstocked, params.half_saturation_K, params.shape_S)

            features.append(saturated)

        # Seasonality feature: sine wave with yearly period
        week_numbers = df["week"].values if "week" in df.columns else np.arange(len(df))
        seasonality = np.sin(2 * np.pi * week_numbers / 52.0)
        features.append(seasonality)

        return np.column_stack(features)

    def fit(self, df: pd.DataFrame) -> "MarketingMixModel":
        """Fit the MMM model on historical data.

        Args:
            df: DataFrame with columns: week, spend_<channel>, revenue.

        Returns:
            Self (fitted model).
        """
        self._train_df = df.copy()
        X = self._build_features(df)
        y = df["revenue"].values

        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X, y)

        # Store coefficients
        for i, ch in enumerate(self.channels):
            self.coefficients[ch] = float(self.model.coef_[i])

        self._seasonality_coef = float(self.model.coef_[-1])
        self.intercept = float(self.model.intercept_)

        self._X_transformed = X
        self._y_true = y
        self._y_pred = self.model.predict(X)
        self._fitted = True

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict revenue from investment data.

        Args:
            df: DataFrame with spend columns per channel.

        Returns:
            Predicted revenue array.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        X = self._build_features(df)
        return self.model.predict(X)

    def get_metrics(self) -> Dict[str, float]:
        """Compute model quality metrics.

        Returns:
            Dictionary with R2, adjusted R2, MAPE, MAE, NRMSE.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")

        n = len(self._y_true)
        p = self._X_transformed.shape[1]
        r2 = r2_score(self._y_true, self._y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # MAPE (avoiding division by zero)
        mask = self._y_true != 0
        mape = float(np.mean(np.abs((self._y_true[mask] - self._y_pred[mask]) / self._y_true[mask])) * 100)

        mae = float(mean_absolute_error(self._y_true, self._y_pred))
        rmse = float(np.sqrt(np.mean((self._y_true - self._y_pred) ** 2)))
        nrmse = rmse / float(np.mean(self._y_true)) * 100

        return {
            "r_squared": round(r2, 4),
            "adjusted_r_squared": round(adj_r2, 4),
            "mape_pct": round(mape, 2),
            "mae": round(mae, 2),
            "nrmse_pct": round(nrmse, 2),
            "n_observations": n,
            "n_features": p,
        }

    def decompose(self) -> Dict:
        """Decompose total revenue into base organic + channel contributions + seasonality.

        Returns:
            Dictionary with total contributions per channel and weekly breakdown.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")

        df = self._train_df
        X = self._X_transformed
        n_channels = len(self.channels)

        weekly_data = []
        channel_totals = {ch: 0.0 for ch in self.channels}

        for t in range(len(df)):
            row = {"week": int(df.iloc[t].get("week", t + 1))}
            row["base_organic"] = self.intercept
            row["seasonality"] = float(X[t, -1] * self._seasonality_coef)

            for i, ch in enumerate(self.channels):
                contribution = float(X[t, i] * self.coefficients[ch])
                row[ch] = max(contribution, 0.0)
                channel_totals[ch] += max(contribution, 0.0)

            row["total_predicted"] = float(self._y_pred[t])
            row["total_actual"] = float(self._y_true[t])
            weekly_data.append(row)

        total_predicted = float(np.sum(self._y_pred))
        total_actual = float(np.sum(self._y_true))
        base_total = self.intercept * len(df)

        channel_pcts = {}
        for ch in self.channels:
            channel_pcts[ch] = round(channel_totals[ch] / total_predicted * 100, 2) if total_predicted > 0 else 0.0

        return {
            "total_actual_revenue": round(total_actual, 2),
            "total_predicted_revenue": round(total_predicted, 2),
            "base_organic_total": round(base_total, 2),
            "base_organic_pct": round(base_total / total_predicted * 100, 2) if total_predicted > 0 else 0.0,
            "channel_contributions": {ch: round(v, 2) for ch, v in channel_totals.items()},
            "channel_contribution_pcts": channel_pcts,
            "weekly_data": weekly_data,
        }

    def simulate_scenario(self, budget_allocation: Dict[str, float]) -> Dict:
        """Simulate a what-if scenario with custom budget allocation.

        Args:
            budget_allocation: Weekly spend per channel.

        Returns:
            Projected revenue and per-channel metrics.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")

        total_budget = sum(budget_allocation.values())
        projected_revenue = self.intercept

        channels_detail = []
        for ch in self.channels:
            spend = budget_allocation.get(ch, 0.0)
            params = settings.channels[ch]

            # Apply Adstock (steady-state approximation for single-week simulation)
            effective_spend = spend / (1.0 - params.decay_rate) if params.decay_rate < 1.0 else spend

            # Apply Hill saturation
            saturated = float(apply_hill_saturation(
                np.array([effective_spend]), params.half_saturation_K, params.shape_S
            )[0])

            beta = self.coefficients.get(ch, 0.0)
            attributed = saturated * beta
            attributed = max(attributed, 0.0)
            projected_revenue += attributed

            roas = attributed / spend if spend > 0 else 0.0
            mroas = compute_marginal_roas(effective_spend, params.half_saturation_K, params.shape_S, beta)
            health = classify_channel_health(mroas)

            channels_detail.append({
                "channel": ch,
                "display_name": dict(zip(
                    ["meta_ads", "google_ads", "tiktok_ads", "influencers", "crm_email"],
                    ["Meta Ads (Facebook & Instagram)", "Google Ads (Search & Display)",
                     "TikTok Ads", "Influenciadores & Creators", "CRM & E-mail Marketing"]
                )).get(ch, ch),
                "spend": round(spend, 2),
                "attributed_revenue": round(attributed, 2),
                "roas": round(roas, 2),
                "marginal_roas": round(mroas, 4),
                "contribution_pct": 0.0,
                "health": health,
            })

        # Calculate contribution percentages
        media_total = sum(c["attributed_revenue"] for c in channels_detail)
        for c in channels_detail:
            c["contribution_pct"] = round(
                c["attributed_revenue"] / media_total * 100, 2
            ) if media_total > 0 else 0.0

        roas_global = projected_revenue / total_budget if total_budget > 0 else 0.0

        return {
            "total_weekly_budget": round(total_budget, 2),
            "projected_weekly_revenue": round(projected_revenue, 2),
            "roas_global": round(roas_global, 2),
            "base_organic_revenue": round(self.intercept, 2),
            "channels": channels_detail,
        }

    def get_response_curves(self, n_points: int = 50) -> List[Dict]:
        """Generate response curve data for each channel.

        Args:
            n_points: Number of points to sample along the spend axis.

        Returns:
            List of channel response curve data.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first.")

        curves = []
        for ch in self.channels:
            params = settings.channels[ch]
            beta = self.coefficients.get(ch, 0.0)
            current_spend = params.weekly_budget_default

            max_spend = current_spend * 3.0
            spends = np.linspace(0, max_spend, n_points)

            points = []
            for s in spends:
                effective = s / (1.0 - params.decay_rate) if params.decay_rate < 1.0 else s
                saturated = float(apply_hill_saturation(
                    np.array([effective]), params.half_saturation_K, params.shape_S
                )[0])
                incremental = max(saturated * beta, 0.0)
                mroas = compute_marginal_roas(effective, params.half_saturation_K, params.shape_S, beta)

                points.append({
                    "spend": round(float(s), 2),
                    "incremental_revenue": round(incremental, 2),
                    "marginal_roas": round(mroas, 4),
                })

            # Saturation point: where marginal ROAS drops below 1.0
            sat_point = max_spend
            for p in points:
                if p["marginal_roas"] < 1.0 and p["spend"] > 0:
                    sat_point = p["spend"]
                    break

            from src.domain.models import CHANNEL_DISPLAY_NAMES
            curves.append({
                "channel": ch,
                "display_name": CHANNEL_DISPLAY_NAMES.get(ch, ch),
                "current_spend": round(current_spend, 2),
                "saturation_point": round(sat_point, 2),
                "points": points,
            })

        return curves

"""Constrained budget optimizer using SciPy SLSQP.

Finds the optimal allocation of a total marketing budget across channels
that maximizes projected revenue, subject to per-channel min/max constraints.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional

from src.engine.transformations import apply_hill_saturation, compute_marginal_roas, classify_channel_health
from src.config.settings import settings


class BudgetOptimizer:
    """Constrained multi-channel budget allocation optimizer."""

    def __init__(self, model):
        """Initialize optimizer with a fitted MMM model.

        Args:
            model: A fitted MarketingMixModel instance.
        """
        self.model = model
        self.channels = list(settings.channels.keys())

    def _objective(self, allocations: np.ndarray) -> float:
        """Negative projected revenue (to minimize = maximize revenue).

        Args:
            allocations: Array of spend values per channel.

        Returns:
            Negative total revenue (for minimization).
        """
        total_revenue = self.model.intercept

        for i, ch in enumerate(self.channels):
            spend = allocations[i]
            params = settings.channels[ch]
            beta = self.model.coefficients.get(ch, 0.0)

            # Steady-state Adstock approximation
            effective = spend / (1.0 - params.decay_rate) if params.decay_rate < 1.0 else spend

            saturated = float(apply_hill_saturation(
                np.array([effective]), params.half_saturation_K, params.shape_S
            )[0])

            total_revenue += max(saturated * beta, 0.0)

        return -total_revenue

    def optimize(
        self,
        total_budget: float,
        min_per_channel: Optional[Dict[str, float]] = None,
        max_per_channel: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Find optimal budget allocation maximizing projected revenue.

        Args:
            total_budget: Total weekly budget to allocate in BRL.
            min_per_channel: Optional minimum spend per channel.
            max_per_channel: Optional maximum spend per channel.

        Returns:
            Optimization result with optimal allocation and projected metrics.
        """
        min_per_channel = min_per_channel or {}
        max_per_channel = max_per_channel or {}

        n = len(self.channels)

        # Initial guess: uniform allocation
        x0 = np.full(n, total_budget / n)

        # Bounds per channel
        bounds = []
        for ch in self.channels:
            lb = min_per_channel.get(ch, 0.0)
            ub = max_per_channel.get(ch, total_budget)
            bounds.append((lb, ub))

        # Equality constraint: sum of allocations = total budget
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
        ]

        result = minimize(
            self._objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        optimal_allocation = {ch: round(float(result.x[i]), 2) for i, ch in enumerate(self.channels)}
        optimal_revenue = -result.fun

        # Compute uniform baseline for lift calculation
        uniform_alloc = {ch: total_budget / n for ch in self.channels}
        uniform_result = self.model.simulate_scenario(uniform_alloc)
        uniform_revenue = uniform_result["projected_weekly_revenue"]

        lift_pct = ((optimal_revenue - uniform_revenue) / uniform_revenue * 100) if uniform_revenue > 0 else 0.0

        # Build per-channel detail
        from src.domain.models import CHANNEL_DISPLAY_NAMES
        channels_detail = []
        media_total = 0.0

        for ch in self.channels:
            spend = optimal_allocation[ch]
            params = settings.channels[ch]
            beta = self.model.coefficients.get(ch, 0.0)
            effective = spend / (1.0 - params.decay_rate) if params.decay_rate < 1.0 else spend

            saturated = float(apply_hill_saturation(
                np.array([effective]), params.half_saturation_K, params.shape_S
            )[0])
            attributed = max(saturated * beta, 0.0)
            media_total += attributed

            roas = attributed / spend if spend > 0 else 0.0
            mroas = compute_marginal_roas(effective, params.half_saturation_K, params.shape_S, beta)
            health = classify_channel_health(mroas)

            channels_detail.append({
                "channel": ch,
                "display_name": CHANNEL_DISPLAY_NAMES.get(ch, ch),
                "spend": round(spend, 2),
                "attributed_revenue": round(attributed, 2),
                "roas": round(roas, 2),
                "marginal_roas": round(mroas, 4),
                "contribution_pct": 0.0,
                "health": health,
            })

        for c in channels_detail:
            c["contribution_pct"] = round(
                c["attributed_revenue"] / media_total * 100, 2
            ) if media_total > 0 else 0.0

        roas_global = optimal_revenue / total_budget if total_budget > 0 else 0.0

        return {
            "total_budget": round(total_budget, 2),
            "optimal_allocation": optimal_allocation,
            "projected_revenue": round(optimal_revenue, 2),
            "roas_global": round(roas_global, 2),
            "lift_vs_uniform_pct": round(lift_pct, 2),
            "channels": channels_detail,
        }

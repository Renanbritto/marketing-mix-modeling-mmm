"""Application settings and default econometric parameters."""

from typing import Dict
from pydantic import BaseModel, Field


class ChannelDefaults(BaseModel):
    """Default Adstock and Hill saturation parameters per media channel."""
    decay_rate: float = Field(..., ge=0.0, le=1.0, description="Adstock geometric decay rate (0=no memory, 1=full memory)")
    half_saturation_K: float = Field(..., gt=0.0, description="Hill half-saturation point (spend level at 50% effect)")
    shape_S: float = Field(..., gt=0.0, description="Hill shape/steepness parameter")
    weekly_budget_default: float = Field(..., ge=0.0, description="Default weekly spend in BRL")


class AppSettings(BaseModel):
    """Global configuration for the Marketing Mix Modeling system."""
    app_name: str = "MMM Budget Optimizer - Motor de Marketing Mix Modeling"
    app_version: str = "1.0.0"
    environment: str = "production"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    random_seed: int = 42

    # Simulation parameters
    n_weeks: int = Field(default=104, description="Number of weeks of historical data to generate")
    base_revenue_weekly: float = Field(default=180_000.0, description="Baseline organic weekly revenue in BRL")

    # Channel configurations with realistic Brazilian market parameters
    channels: Dict[str, ChannelDefaults] = Field(default_factory=lambda: {
        "meta_ads": ChannelDefaults(
            decay_rate=0.65,
            half_saturation_K=45_000.0,
            shape_S=1.8,
            weekly_budget_default=38_000.0,
        ),
        "google_ads": ChannelDefaults(
            decay_rate=0.35,
            half_saturation_K=35_000.0,
            shape_S=2.2,
            weekly_budget_default=32_000.0,
        ),
        "tiktok_ads": ChannelDefaults(
            decay_rate=0.50,
            half_saturation_K=20_000.0,
            shape_S=1.5,
            weekly_budget_default=15_000.0,
        ),
        "influencers": ChannelDefaults(
            decay_rate=0.80,
            half_saturation_K=25_000.0,
            shape_S=1.3,
            weekly_budget_default=12_000.0,
        ),
        "crm_email": ChannelDefaults(
            decay_rate=0.20,
            half_saturation_K=8_000.0,
            shape_S=2.5,
            weekly_budget_default=5_000.0,
        ),
    })


settings = AppSettings()

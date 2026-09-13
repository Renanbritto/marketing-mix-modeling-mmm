"""Pydantic schemas for API inputs, outputs and analytical serialization."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Input for what-if budget simulation scenarios."""
    budget_allocation: Dict[str, float] = Field(
        ...,
        description="Weekly spend per channel in BRL (e.g. {'meta_ads': 40000, 'google_ads': 30000})"
    )


class ChannelDetail(BaseModel):
    """Per-channel breakdown in simulation/optimization results."""
    channel: str
    display_name: str
    spend: float
    attributed_revenue: float
    roas: float
    marginal_roas: float
    contribution_pct: float
    health: str


class SimulationResponse(BaseModel):
    """Output of a budget simulation scenario."""
    total_weekly_budget: float
    projected_weekly_revenue: float
    roas_global: float
    base_organic_revenue: float
    channels: List[ChannelDetail]


class OptimizationRequest(BaseModel):
    """Input for constrained budget optimization."""
    total_budget: float = Field(..., gt=0, description="Total weekly budget to allocate in BRL")
    min_per_channel: Dict[str, float] = Field(
        default_factory=dict,
        description="Minimum spend per channel (optional)"
    )
    max_per_channel: Dict[str, float] = Field(
        default_factory=dict,
        description="Maximum spend per channel (optional)"
    )


class OptimizationResponse(BaseModel):
    """Output of constrained budget optimization."""
    total_budget: float
    optimal_allocation: Dict[str, float]
    projected_revenue: float
    roas_global: float
    lift_vs_uniform_pct: float
    channels: List[ChannelDetail]


class DecompositionWeek(BaseModel):
    """Revenue decomposition for a single week."""
    week: int
    base_organic: float
    seasonality: float
    meta_ads: float = 0.0
    google_ads: float = 0.0
    tiktok_ads: float = 0.0
    influencers: float = 0.0
    crm_email: float = 0.0
    total_predicted: float
    total_actual: float


class DecompositionResponse(BaseModel):
    """Full revenue decomposition across all weeks."""
    total_actual_revenue: float
    total_predicted_revenue: float
    base_organic_total: float
    base_organic_pct: float
    channel_contributions: Dict[str, float]
    channel_contribution_pcts: Dict[str, float]
    weekly_data: List[DecompositionWeek]


class ResponseCurvePoint(BaseModel):
    """Single point on a channel response curve."""
    spend: float
    incremental_revenue: float
    marginal_roas: float


class ResponseCurveData(BaseModel):
    """Response curve data for a single channel."""
    channel: str
    display_name: str
    current_spend: float
    saturation_point: float
    points: List[ResponseCurvePoint]


class ModelMetricsResponse(BaseModel):
    """Model quality and fit metrics."""
    r_squared: float
    adjusted_r_squared: float
    mape_pct: float
    mae: float
    nrmse_pct: float
    n_observations: int
    n_features: int


class ChannelSummary(BaseModel):
    """Summary health report for a media channel."""
    channel: str
    display_name: str
    total_spend: float
    attributed_revenue: float
    roas: float
    marginal_roas: float
    contribution_pct: float
    health: str
    recommendation: str


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    app_name: str
    version: str
    uptime_status: str

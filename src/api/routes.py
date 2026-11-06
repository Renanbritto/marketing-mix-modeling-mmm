"""API route definitions for the Marketing Mix Modeling system."""

from typing import List, Dict
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.domain.schemas import (
    SimulationRequest,
    SimulationResponse,
    OptimizationRequest,
    OptimizationResponse,
    DecompositionResponse,
    ModelMetricsResponse,
    ResponseCurveData,
    ChannelSummary,
    HealthResponse,
)
from src.domain.models import CHANNEL_DISPLAY_NAMES
from src.engine.mmm_model import MarketingMixModel
from src.engine.optimizer import BudgetOptimizer
from src.data.sample_generator import MarketingDataGenerator
from src.config.settings import settings

router = APIRouter()

# Global state for model and optimizer (lazy loaded/initialized)
_model: MarketingMixModel = None
_optimizer: BudgetOptimizer = None


def get_model() -> MarketingMixModel:
    """Get or initialize the trained MMM model."""
    global _model, _optimizer
    if _model is None:
        generator = MarketingDataGenerator(seed=settings.random_seed)
        df = generator.generate(n_weeks=settings.n_weeks)
        _model = MarketingMixModel(alpha=1.0)
        _model.fit(df)
        _optimizer = BudgetOptimizer(_model)
    return _model


def get_optimizer() -> BudgetOptimizer:
    """Get or initialize the budget optimizer."""
    global _optimizer
    if _optimizer is None:
        get_model()
    return _optimizer


@router.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def get_dashboard(request: Request):
    """Serve the interactive MMM analytics and budget optimization dashboard."""
    templates = getattr(request.app.state, "templates", None)
    if templates is None:
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="src/templates")
    
    model = get_model()
    metrics = model.get_metrics()
    decomp = model.decompose()
    
    # Calculate global totals
    total_revenue = decomp["total_actual_revenue"]
    total_budget = sum(
        settings.channels[ch].weekly_budget_default * settings.n_weeks 
        for ch in settings.channels
    )
    global_roas = total_revenue / total_budget if total_budget > 0 else 0.0

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "metrics": metrics,
            "total_revenue": total_revenue,
            "total_budget": total_budget,
            "global_roas": global_roas,
            "channels": settings.channels,
            "channel_names": CHANNEL_DISPLAY_NAMES,
            "settings": settings,
            "n_weeks": settings.n_weeks,
        },
    )


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for container orchestrators and monitoring."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        uptime_status="operational",
    )


@router.get("/api/v1/model/metrics", response_model=ModelMetricsResponse, tags=["Econometric Engine"])
async def get_model_metrics():
    """Retrieve econometric model quality, goodness-of-fit and error metrics."""
    model = get_model()
    return model.get_metrics()


@router.get("/api/v1/model/decomposition", response_model=DecompositionResponse, tags=["Econometric Engine"])
async def get_model_decomposition():
    """Retrieve full revenue decomposition across baseline, media channels, and seasonality."""
    model = get_model()
    return model.decompose()


@router.get("/api/v1/model/response-curves", response_model=List[ResponseCurveData], tags=["Econometric Engine"])
async def get_response_curves():
    """Retrieve Hill response curves and marginal saturation points for each media channel."""
    model = get_model()
    return model.get_response_curves()


@router.post("/api/v1/simulate", response_model=SimulationResponse, tags=["Simulation & Optimization"])
async def simulate_scenario(payload: SimulationRequest):
    """Simulate a what-if weekly budget allocation scenario across channels."""
    model = get_model()
    try:
        result = model.simulate_scenario(payload.budget_allocation)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/v1/optimize", response_model=OptimizationResponse, tags=["Simulation & Optimization"])
async def optimize_budget(payload: OptimizationRequest):
    """Find the mathematically optimal weekly budget allocation to maximize revenue."""
    optimizer = get_optimizer()
    try:
        result = optimizer.optimize(
            total_budget=payload.total_budget,
            min_per_channel=payload.min_per_channel,
            max_per_channel=payload.max_per_channel,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/v1/channels/summary", response_model=List[ChannelSummary], tags=["Channel Analytics"])
async def get_channels_summary():
    """Retrieve an executive summary of performance, ROAS, and health status per channel."""
    model = get_model()
    decomp = model.decompose()
    curves = model.get_response_curves()
    curve_map = {c["channel"]: c for c in curves}

    summaries = []
    for ch, params in settings.channels.items():
        spend = params.weekly_budget_default * settings.n_weeks
        attributed = decomp["channel_contributions"].get(ch, 0.0)
        roas = attributed / spend if spend > 0 else 0.0
        
        # Marginal ROAS from response curve at current spend
        sat_point = curve_map.get(ch, {}).get("saturation_point", 0.0)
        current_spend = params.weekly_budget_default
        beta = model.coefficients.get(ch, 0.0)
        effective = current_spend / (1.0 - params.decay_rate) if params.decay_rate < 1.0 else current_spend
        
        from src.engine.transformations import compute_marginal_roas, classify_channel_health
        mroas = compute_marginal_roas(effective, params.half_saturation_K, params.shape_S, beta)
        health = classify_channel_health(mroas)

        recommendation_map = {
            "SCALING": "Canal altamente eficiente com espaço para aceleração imediata de investimento.",
            "EFFICIENT": "Retorno balanceado. Manter investimento ou aumentar moderadamente.",
            "SATURATED": "Retornos decrescentes evidentes. Evitar aumento de budget sem otimização de criativos.",
            "OVER_INVESTED": "Canal em saturação crítica (mROAS < 0.8). Recomenda-se realocar budget para canais mais eficientes.",
        }

        summaries.append(
            ChannelSummary(
                channel=ch,
                display_name=CHANNEL_DISPLAY_NAMES.get(ch, ch),
                total_spend=round(spend, 2),
                attributed_revenue=round(attributed, 2),
                roas=round(roas, 2),
                marginal_roas=round(mroas, 4),
                contribution_pct=round(decomp["channel_contribution_pcts"].get(ch, 0.0), 2),
                health=health,
                recommendation=recommendation_map.get(health, "Monitorar métricas de conversão."),
            )
        )

    return summaries

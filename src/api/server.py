"""FastAPI application factory and middleware configuration."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config.settings import settings
from src.api.routes import router, get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: preheats model and resources on startup."""
    # Pre-train MMM model on startup so initial requests are instantaneous
    get_model()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API para Marketing Mix Modeling (MMM) com transformações Adstock, "
            "curvas de saturação Hill e otimização de orçamento multicanal via SciPy SLSQP."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount templates and static directory if available
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(base_dir, "templates")
    static_dir = os.path.join(base_dir, "static")

    if os.path.exists(templates_dir):
        app.state.templates = Jinja2Templates(directory=templates_dir)

    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Register routes
    app.include_router(router)

    return app

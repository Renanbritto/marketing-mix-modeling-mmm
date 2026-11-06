"""FastAPI application factory and middleware configuration."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.types import ASGIApp, Scope, Receive, Send

from src.config.settings import settings
from src.api.routes import router, get_model


class VercelPathMiddleware:
    """ASGI Middleware to ensure accurate route matching on Vercel Serverless Functions.

    When Vercel rewrites requests to /api/index.py, the ASGI scope['path'] can be set
    to '/api/index.py' instead of the original request path. Vercel forwards the original
    requested URI in the 'x-matched-path' or 'x-forwarded-uri' HTTP headers.
    This middleware restores the original path so FastAPI routes match cleanly.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            headers = dict(scope.get("headers", []))
            raw_matched = (
                headers.get(b"x-matched-path", b"").decode("latin1")
                or headers.get(b"x-forwarded-uri", b"").decode("latin1")
            )
            matched_path = raw_matched.split("?")[0] if raw_matched else ""

            if matched_path and matched_path.startswith("/") and not matched_path.startswith("/api/index.py"):
                scope["path"] = matched_path
            elif path in ("/api/index.py", "/api", "/api/", "/api/index"):
                scope["path"] = "/"
            elif path.startswith("/api/index.py/"):
                scope["path"] = path[len("/api/index.py") :]

        await self.app(scope, receive, send)


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

    # Vercel ASGI path normalizer
    app.add_middleware(VercelPathMiddleware)

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

"""Vercel Serverless Function entrypoint."""

import sys
import os
import traceback
import json

# Ensure the repository root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

_init_error = None
raw_app = None

try:
    from src.api.server import create_app
    raw_app = create_app()
except Exception as exc:
    _init_error = traceback.format_exc()
    print(f"[FATAL STARTUP] {exc}\n{_init_error}", file=sys.stderr)


async def app(scope, receive, send):
    """Resilient ASGI entrypoint for Vercel Serverless Functions."""
    if _init_error:
        body = json.dumps({"error": "Startup initialization error", "traceback": _init_error}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode("ascii")],
            ],
        })
        await send({"type": "http.response.body", "body": body})
        return

    if scope.get("type") == "lifespan":
        while True:
            message = await receive()
            if message.get("type") == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message.get("type") == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    try:
        await raw_app(scope, receive, send)
    except Exception as exc:
        tb = traceback.format_exc()
        body = json.dumps({"error": "Runtime invocation error", "detail": str(exc), "traceback": tb}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode("ascii")],
            ],
        })
        await send({"type": "http.response.body", "body": body})

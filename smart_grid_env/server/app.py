"""
FastAPI Server Entry Point
===========================
Exposes the SmartGridEnv over HTTP/WebSocket for the OpenEnv CLI and agents.
"""

import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Force web interface to be enabled
os.environ["ENABLE_WEB_INTERFACE"] = "true"

from openenv.core.env_server.http_server import create_app
from .grid_env import SmartGridEnv
from models import Action, Observation


# Create the FastAPI app using the OpenEnv helper
# This automatically creates /reset, /step, /state, /ws, and /web endpoints
app = create_app(
    SmartGridEnv,
    Action,
    Observation,
    env_name="smart-grid-demand-response",
    max_concurrent_envs=10,  # Allow multiple parallel sessions for training
)


@app.get("/")
async def root_redirect():
    """Redirect root to the interactive web interface."""
    return RedirectResponse(url="/web/")


def main():
    """Entry point for the server."""
    import uvicorn
    import sys
    
    port = int(os.environ.get("PORT", 7860))
    if "--port" in sys.argv:
        p_idx = sys.argv.index("--port")
        if p_idx + 1 < len(sys.argv):
            port = int(sys.argv[p_idx + 1])
            
    print(f"--- Starting Smart Grid Demand Response Server on port {port} ---")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

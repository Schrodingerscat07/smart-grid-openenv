# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Smart Grid Environment.

This module creates an HTTP server that exposes the SmartGridEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
os.environ["ENABLE_WEB_INTERFACE"] = "true"

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SmartGridAction, SmartGridObservation
    from .smart_grid_environment import SmartGridEnvironment
except ModuleNotFoundError:
    from models import SmartGridAction, SmartGridObservation
    from server.smart_grid_environment import SmartGridEnvironment


# Create the app with web interface and README integration
app = create_app(
    SmartGridEnvironment,
    SmartGridAction,
    SmartGridObservation,
    env_name="smart_grid",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main():
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    import sys
    
    # Simple argparse for port
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    elif "--port" in sys.argv:
        p_idx = sys.argv.index("--port")
        if p_idx + 1 < len(sys.argv) and sys.argv[p_idx + 1].isdigit():
            port = int(sys.argv[p_idx + 1])

    print(f"--- Starting server on PORT {port} ---")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

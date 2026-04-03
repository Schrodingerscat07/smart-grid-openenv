import os
# Force web interface to reach into the right code
os.environ["ENABLE_WEB_INTERFACE"] = "true"

from openenv.core.env_server.http_server import create_app
from .grid_env import SmartGridEnv
from models import Action, Observation

# PASS THE CLASS (not an instance)
app = create_app(
    SmartGridEnv, 
    Action,
    Observation,
    env_name="smart_grid_demand_response"
)

# Also redirect the root to /web/ to make it easier to find
from fastapi.responses import RedirectResponse
@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

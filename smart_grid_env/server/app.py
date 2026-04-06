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

import gradio as gr

def custom_gradio_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Adds a custom tab to the web UI allowing task selection during reset."""
    with gr.Blocks() as custom:
        gr.Markdown("### Task Configuration\nSelect a task setting below and hit **Initialize Task**. This updates the backend state for the entire interface, so you can switch back to the **Playground** tab to view your situation report and take steps!")
        
        with gr.Row():
            task_choice = gr.Dropdown(
                choices=["peak_survival", "daily_balance", "extreme_event"],
                value="peak_survival",
                label="Select Scenario Task"
            )
            reset_btn = gr.Button("Initialize Task", variant="primary")
            
        status = gr.Textbox(label="Initialization Status")
            
        async def do_reset(task):
            try:
                import asyncio
                from openenv.core.env_server.serialization import serialize_observation
                # Call env.reset directly with task_name (bypasses reset_environment's limited signature)
                loop = asyncio.get_event_loop()
                observation = await loop.run_in_executor(
                    None, lambda: web_manager.env.reset(task_name=task)
                )
                serialized = serialize_observation(observation)
                # Sync the web manager's internal episode state
                web_manager.episode_state.episode_id = web_manager.env.state.episode_id
                web_manager.episode_state.step_count = 0
                web_manager.episode_state.current_observation = serialized["observation"]
                web_manager.episode_state.action_logs = []
                web_manager.episode_state.is_reset = True
                return f"✅ Successfully loaded '{task}'. Switch back to the Playground tab to begin!"
            except Exception as e:
                return f"❌ Error: {str(e)}"
                
        reset_btn.click(fn=do_reset, inputs=[task_choice], outputs=[status])
        
    return custom


# Create the FastAPI app using the OpenEnv helper
# This automatically creates /reset, /step, /state, /ws, and /web endpoints
app = create_app(
    SmartGridEnv,
    Action,
    Observation,
    env_name="smart-grid-demand-response",
    max_concurrent_envs=10,  # Allow multiple parallel sessions for training
    gradio_builder=custom_gradio_ui,
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

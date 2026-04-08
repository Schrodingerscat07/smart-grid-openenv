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
    """Adds a beautiful Control Room dashboard to the web UI."""
    def strip_frontmatter(text: str) -> str:
        if not text: return ""
        import re
        # Find the first occurrence of --- block at the start (ignoring leading whitespace/BOM)
        match = re.search(r'^\s*---\s*$.*?^\s*---\s*$(.*)', text, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return text.strip()

    clean_readme = strip_frontmatter(metadata.readme_content or "")
    
    with gr.Blocks() as custom:
        gr.Markdown(clean_readme)
        gr.Markdown("---")
        gr.Markdown("## 🎮 Smart Grid Control Room\nThis is your main dashboard. The situation report below updates dynamically after every action.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # ── The main Situation Report display ──
                situation_md = gr.Markdown(
                    "*(Initialize a task to view the Situation Report)*", 
                    elem_id="situation-report",
                    line_breaks=True
                )
                
            with gr.Column(scale=1):
                # ── Controls ──
                gr.Markdown("### 1. Scenario Setup")
                task_choice = gr.Dropdown(
                    choices=["peak_survival", "daily_balance", "extreme_event", "monsoon_crisis", "renewable_transition"],
                    value="peak_survival",
                    label="Select Scenario Task"
                )
                init_btn = gr.Button("Initialize Task / Reset", variant="secondary")
                
                gr.Markdown("---")
                gr.Markdown("### 2. Actions")
                curtail_input = gr.Code(
                    label="Curtailments (JSON)", 
                    language="json", 
                    value='{\n  "steel_plant": 0.0\n}'
                )
                bat_action = gr.Radio(["idle", "charge", "discharge"], value="idle", label="Battery Action")
                bat_mw = gr.Slider(0, 25, value=0, step=1, label="Battery MW")
                
                step_btn = gr.Button("Execute Step (1 Hour)", variant="primary", size="lg")
                
                gr.Markdown("---")
                gr.Markdown("### 3. Episode Status")
                status_box = gr.Textbox(label="System Messages", value="Ready.")
                grade_box = gr.Number(label="Final Grade", visible=False)

        async def do_init(task):
            try:
                import asyncio
                from openenv.core.env_server.serialization import serialize_observation
                loop = asyncio.get_event_loop()
                obs = await loop.run_in_executor(None, lambda: web_manager.env.reset(task_name=task))
                # Sync web manager
                web_manager.episode_state.episode_id = web_manager.env.state.episode_id
                web_manager.episode_state.step_count = 0
                web_manager.episode_state.current_observation = serialize_observation(obs)["observation"]
                web_manager.episode_state.action_logs = []
                web_manager.episode_state.is_reset = True
                
                report = obs.situation_report
                return f"✅ Loaded '{task}'.", report, gr.update(visible=False)
            except Exception as e:
                return f"❌ Error: {str(e)}", "Error loading report.", gr.update(visible=False)
                
        async def do_step(curtails_json, b_action, b_mw):
            try:
                import json
                import asyncio
                from openenv.core.env_server.serialization import serialize_observation
                
                c_dict = json.loads(curtails_json) if curtails_json.strip() else {}
                action = Action(curtailments=c_dict, battery_action=b_action, battery_mw=b_mw)
                
                loop = asyncio.get_event_loop()
                obs = await loop.run_in_executor(None, lambda: web_manager.env.step(action))
                
                # Sync web manager
                web_manager.episode_state.step_count = web_manager.env.state.step_count
                web_manager.episode_state.current_observation = serialize_observation(obs)["observation"]
                
                report = obs.situation_report
                status = f"Step {obs.step_number} executed. Reward: {obs.reward:.3f}"
                
                grade_update = gr.update(visible=False)
                if obs.done:
                    final_grade = await loop.run_in_executor(None, lambda: web_manager.env.grade())
                    status = f"🛑 Episode Complete. Grade: {final_grade:.4f}"
                    grade_update = gr.update(value=final_grade, visible=True)
                    
                return status, report, grade_update
            except Exception as e:
                return f"❌ Error: {str(e)}", "Error.", gr.update(visible=False)

        init_btn.click(fn=do_init, inputs=[task_choice], outputs=[status_box, situation_md, grade_box])
        step_btn.click(fn=do_step, inputs=[curtail_input, bat_action, bat_mw], outputs=[status_box, situation_md, grade_box])
        
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

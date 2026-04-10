"""
Baseline LLM Agent Inference (Phase 2)
========================================
A simple baseline that uses an LLM to control the grid.
Uses the 'situation_report' from observations for zero-shot reasoning.

Phase 2 update:
- Strictly adheres to mandatory STDOUT logging format ([START], [STEP], [END]).
"""

import os
import json
from openai import OpenAI
from server.grid_env import SmartGridEnv
from models import Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

# Configure the client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def run_episode(task_name: str) -> float:
    """Run one full episode with the LLM agent using strict STDOUT logging."""
    env = SmartGridEnv()
    obs = env.reset(task_name=task_name)
    
    rewards = []
    step_count = 0
    success = False
    
    # [START] mandatory stdout
    print(f"[START] task={task_name} env=smart-grid-demand-response model={MODEL}", flush=True)

    while True:
        # Build prompt from the situation_report
        loads_summary = json.dumps(
            [
                {
                    "id": l["id"],
                    "current_mw": l["current_mw"],
                    "max_reducible_mw": l["reducible_mw"],
                    "priority": l["priority"],
                    "tripped": l.get("tripped", False),
                }
                for l in obs.loads
            ]
        )

        system_prompt = """You are a Master AI Smart Grid Dispatcher.
Your goal is to maximize stability (50Hz) while minimizing cost and discomfort.
RULES:
1. If frequency < 49.8Hz, supply is critically low. You MUST curtail low/medium priority loads (steel_plant, cement_factory) or discharge the battery.
2. If frequency > 50.1Hz, supply is high. You MUST charge the battery to store the surplus.
3. NEVER curtail critical loads (hospital, metro_rail) unless absolutely necessary to avoid blackout (48.5Hz).
4. Battery max rate is 25MW.
Respond ONLY in valid JSON. Example:
{"curtailments": {"steel_plant": 15.0}, "battery_action": "discharge", "battery_mw": 20.0}"""

        user_prompt = f"""=== CURRENT STATE ===
SITUATION: {obs.situation_report}
GRID FREQ: {obs.grid_frequency_hz} Hz
BATTERY SOC: {obs.battery_soc_pct:.1f}%
LOADS: {loads_summary}"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            action_dict = json.loads(content)

            if "curtailments" not in action_dict:
                action_dict = {"curtailments": action_dict}
            action_dict.setdefault("battery_action", "idle")
            action_dict.setdefault("battery_mw", 0.0)

            action = Action(**action_dict)
            action_str = json.dumps(action_dict, separators=(",", ":"))
            error_msg = "null"
        except Exception as e:
            action = Action(curtailments={}, battery_action="idle", battery_mw=0.0)
            action_str = "{}"
            error_msg = f'"{str(e)}"'

        # Step the environment
        obs = env.step(action)
        step_count += 1
        
        # Format reward
        reward_val = obs.reward if obs.reward is not None else 0.0
        rewards.append(reward_val)
        
        done_str = "true" if obs.done else "false"

        # [STEP] mandatory stdout
        print(f"[STEP] step={step_count} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}", flush=True)

        if obs.done:
            break

    # Final grade
    final_score = env.grade()
    # Hackathon spec logic for 'success' (commonly score >= threshold, e.g., 0.5)
    success = final_score >= 0.5
    success_str = "true" if success else "false"
    
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    
    # [END] mandatory stdout
    print(f"[END] success={success_str} steps={step_count} score={final_score:.2f} rewards={rewards_str}", flush=True)
    
    return final_score


if __name__ == "__main__":
    tasks = [
        "peak_survival",
        "daily_balance",
        "extreme_event",
        "monsoon_crisis",
        "renewable_transition",
    ]
    for task in tasks:
        try:
            run_episode(task)
        except Exception as e:
            # If a massive exception happens, log standard end to prevent crawler crash
            print(f"[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)

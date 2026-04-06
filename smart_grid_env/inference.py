"""
Baseline LLM Agent Inference
==============================
A simple baseline that uses an LLM (like GPT-3.5 or Claude) to control
the grid. This script showcases how to use the 'situation_report' 
from the environment for zero-shot reasoning.
"""

import os
import json
from openai import OpenAI
from server.grid_env import SmartGridEnv
from models import Action


# Configure the client (pointing to HF Spaces or a local API if needed)
# For the hackathon, this will run against the evaluator's selected model.
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("HF_TOKEN", "dummy_key")
)
MODEL = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")


def run_episode(task_name: str) -> float:
    """Run one full episode with the LLM agent and return the final grade."""
    env = SmartGridEnv()
    obs = env.reset(task_name=task_name)
    total_reward = 0
    step_count = 0

    print(f"--- Starting Episode: {task_name} ---")

    while True:
        # 1. Use the 'situation_report' as the primary prompt context
        prompt = f"""
You are an AI Smart Grid Dispatcher controlling a city power grid in India. 
Your goal is to maintain stability (50Hz), protect critical infrastructure, and minimize costs.

{obs.situation_report}

--- LOADS AVAILABLE TO CURTAIL ---
{json.dumps([{"id": l["id"], "name": l["name"], "max_reducible_mw": l["reducible_mw"], "priority": l["priority"]} for l in obs.loads], indent=2)}

INSTRUCTIONS:
- Analyze the situation. Is there a supply deficit?
- Decide which loads to curtail to bring the grid back to 50Hz.
- Avoid curtailing 'critical' loads unless it's a total emergency.
- Prefer 'low' priority industrial loads first.

Respond ONLY with a JSON object in this format:
{{"curtailments": {{"load_id": mw_to_reduce}}}}

Example: {{"curtailments": {{"steel_plant": 15.0, "shopping_mall_1": 5.2}}}}
"""
        # 2. Get LLM response
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            response_format={ "type": "json_object" } # Ensure JSON if model supports it
        )

        # 3. Parse LLM response into an Action
        try:
            content = response.choices[0].message.content
            action_dict = json.loads(content)
            # Ensure it matches our Action model's expected curtailments dict
            if "curtailments" not in action_dict:
                action_dict = {"curtailments": action_dict}
            action = Action(**action_dict)
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Falling back to no-op action.")
            action = Action(curtailments={})

        # 4. Step the environment
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        step_count += 1
        
        print(f"Step {step_count} | Freq: {obs.grid_frequency_hz:.2f}Hz | Reward: {result.reward:.2f}")

        if result.done:
            break

    # 5. Final grading of the episode
    final_score = env.grade()
    print(f"--- Task {task_name} Finished | Final Grade: {final_score:.3f} ---")
    return final_score


if __name__ == "__main__":
    tasks = ["peak_survival", "daily_balance", "extreme_event"]
    scores = {}
    for task in tasks:
        try:
            scores[task] = run_episode(task)
        except Exception as e:
            print(f"Error running task {task}: {e}")
            scores[task] = 0.0
            
    print("\n--- SUMMARY OF BASELINE SCORES ---")
    for task, score in scores.items():
        print(f"{task.upper():<20} | Score: {score:.3f}")

import os, json
from openai import OpenAI
from server.grid_env import SmartGridEnv
from models import Action

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("HF_TOKEN", "dummy_key")
)
MODEL = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

def run_episode(task_name: str) -> float:
    env = SmartGridEnv(task_name=task_name)
    obs = env.reset()
    total_reward = 0

    for _ in range(12):  # 12 steps per episode
        # Give the LLM the current observation as a prompt
        prompt = f"""
You are controlling a city power grid. Current state:
- Hour: {obs.hour}:00
- Grid frequency: {obs.grid_frequency} Hz (normal=50.0, danger<49.5)
- Demand: {obs.total_demand_mw:.1f} MW
- Available supply: {obs.available_supply_mw:.1f} MW
- Solar: {obs.solar_output_mw:.1f} MW, Wind: {obs.wind_output_mw:.1f} MW

Available loads to curtail:
{json.dumps(obs.loads, indent=2)}

Respond ONLY with a JSON object like:
{{"curtailments": {{"load_id": mw_to_reduce, ...}}}}
Only include loads you want to reduce. Use 0 or omit loads you don't touch.
"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        try:
            content = response.choices[0].message.content
            # Basic cleaning if LLM adds markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            action_dict = json.loads(content)
            action = Action(**action_dict)
        except Exception:
            action = Action(curtailments={})  # No-op on parse failure

        result = env.step(action)
        total_reward += result.reward
        obs = result.observation

        if result.done:
            break

    return env.grade()

if __name__ == "__main__":
    for task in ["peak_survival", "daily_balance", "cost_minimization_fair"]:
        score = run_episode(task)
        print(f"Task: {task} | Score: {score:.3f}")

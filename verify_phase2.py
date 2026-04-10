
from server.grid_env import SmartGridEnv
from models import Action
import json

env = SmartGridEnv()

# Test 1: Cascading failure triggers
obs = env.reset(task_name="extreme_event", seed=42)
print("--- Testing Cascading Failure ---")
for i in range(5):
    obs = env.step(Action(curtailments={}, battery_action="idle", battery_mw=0.0))
    print(f"Step {i:02d}: Freq={obs.grid_frequency_hz:.2f}Hz | Tripped={obs.tripped_loads}")

# Test 2: Battery works
obs = env.reset(task_name="daily_balance", seed=42)
print("\n--- Testing Battery ---")
print(f"Initial SOC: {obs.battery_soc_pct}%")

# Charge
obs = env.step(Action(curtailments={}, battery_action="charge", battery_mw=25.0))
print(f"After charge: {obs.battery_soc_pct}%")

# Discharge
obs = env.step(Action(curtailments={}, battery_action="discharge", battery_mw=25.0))
print(f"After discharge: {obs.battery_soc_pct}%")

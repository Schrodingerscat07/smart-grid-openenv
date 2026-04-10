import asyncio
import json
from server.grid_env import SmartGridEnv
from models import Action

def test_peak_survival_grader():
    env = SmartGridEnv()
    obs = env.reset(task_name="peak_survival")
    
    # Do nothing, see what happens
    for _ in range(12):
        action = Action(curtailments={}, battery_action="idle", battery_mw=0.0)
        obs = env.step(action)
    
    grade = env.grade()
    assert 0.0 <= grade <= 1.0, f"Grade {grade} out of bounds"

def test_battery_physics():
    env = SmartGridEnv()
    obs = env.reset(task_name="daily_balance")
    env.simulator.battery.reset(0.50)
    
    # 1. Charge battery
    action_charge = Action(curtailments={}, battery_action="charge", battery_mw=25.0)
    obs = env.step(action_charge)
    assert env.simulator.battery.soc > 0.50, "Battery should have charged"
    
    # 2. Discharge battery
    soc_after_charge = env.simulator.battery.soc
    action_discharge = Action(curtailments={}, battery_action="discharge", battery_mw=25.0)
    obs = env.step(action_discharge)
    
    assert env.simulator.battery.soc < soc_after_charge, "Battery should have discharged"
    # End SOC < 0.50 due to 90% round trip efficiency loss
    assert env.simulator.battery.soc < 0.50, "Efficiency loss should result in net negative energy"

def test_cascading_failures():
    env = SmartGridEnv()
    # extreme_event has huge deficit
    obs = env.reset(task_name="extreme_event")
    
    num_trips_initial = len(env.simulator.tripped_loads)
    assert num_trips_initial == 0, "Should start with 0 tripped loads"
    
    for _ in range(5):
        action = Action(curtailments={}, battery_action="idle", battery_mw=0.0)
        obs = env.step(action)
        
    num_trips_final = len(env.simulator.tripped_loads)
    assert num_trips_final > 0, "Inactive agent during heatwave should cause cascading failures"

def test_all_tasks_initialize():
    tasks = ["peak_survival", "daily_balance", "extreme_event", "monsoon_crisis", "renewable_transition"]
    env = SmartGridEnv()
    
    for t in tasks:
        obs = env.reset(task_name=t)
        assert obs is not None
        assert not obs.done

if __name__ == "__main__":
    import sys
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("Testing Smart Grid Environment Mechanics...")
    test_all_tasks_initialize()
    print("✅ All tasks initialize.")
    
    test_battery_physics()
    print("✅ Battery physics and efficiency working.")
    
    test_cascading_failures()
    print("✅ Cascading failures trigger successfully during critical events.")
    
    test_peak_survival_grader()
    print("✅ Grader produces valid scores.")
    
    print("🎉 All environment tests passed!")

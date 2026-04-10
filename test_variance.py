import json
import random
from typing import Dict, Any

from server.grid_env import SmartGridEnv
from models import Action

TASKS = ["peak_survival", "daily_balance", "extreme_event", "monsoon_crisis", "renewable_transition"]

def run_policy(task_name: str, policy_type: str) -> float:
    env = SmartGridEnv()
    obs = env.reset(task_name=task_name)
    step_count = 0
    
    while True:
        if policy_type == "do_nothing":
            action = Action(curtailments={}, battery_action="idle", battery_mw=0.0)
            
        elif policy_type == "random":
            curtailments = {}
            for l in obs.loads:
                if random.random() > 0.5:
                    curtailments[l["id"]] = random.uniform(0, l["reducible_mw"])
            batt_actions = ["idle", "charge", "discharge"]
            batt_choice = random.choice(batt_actions)
            action = Action(
                curtailments=curtailments,
                battery_action=batt_choice,
                battery_mw=random.uniform(0, 25.0) if batt_choice != "idle" else 0.0
            )
            
        elif policy_type == "smart_heuristic":
            # Hardcoded logic to act like a decent AI
            curtailments = {}
            freq = obs.grid_frequency_hz
            
            # If frequency is dipping, curtail low-priority loads heavily
            if freq < 49.8:
                for l in obs.loads:
                    if l["priority"] in ["low", "medium"]:
                        curtailments[l["id"]] = l["reducible_mw"]
                        
            # Battery logic
            if freq > 50.1 and obs.battery_soc_pct < 90:
                batt = "charge"
                mw = 20.0
            elif freq < 49.7 and obs.battery_soc_pct > 10:
                batt = "discharge"
                mw = 25.0
            else:
                batt = "idle"
                mw = 0.0
                
            action = Action(curtailments=curtailments, battery_action=batt, battery_mw=mw)

        elif policy_type == "super_smart_heuristic":
            # Perfect Oracle: Calculate exact MW imbalance to reach 50.0Hz
            # freq = 50.0 + (imbalance / 100) * 0.65 -> imbalance = (freq - 50.0) * 100 / 0.65
            # Since imbalance = effective_supply - demand. Positive imbalance = surplus.
            freq = obs.grid_frequency_hz
            mw_needed = (50.0 - freq) * 100 / 0.65
            
            curtailments = {}
            batt = "idle"
            batt_mw = 0.0
            
            # Step 1: Use battery if possible
            if mw_needed > 0: # deficit
                batt = "discharge"
                batt_mw = min(25.0, mw_needed)
                mw_needed -= batt_mw
            elif mw_needed < 0: # surplus
                batt = "charge"
                batt_mw = min(25.0, -mw_needed)
                mw_needed += batt_mw
                
            # Step 2: Curtail loads strictly from low to high priority until mw_needed is met
            if mw_needed > 0:
                for target_prio in ["low", "medium", "high"]: # skip critical
                    for l in obs.loads:
                        if l["priority"] == target_prio:
                            reduce_amount = min(l["reducible_mw"], mw_needed)
                            if reduce_amount > 0:
                                curtailments[l["id"]] = reduce_amount
                                mw_needed -= reduce_amount

            action = Action(curtailments=curtailments, battery_action=batt, battery_mw=batt_mw)

        obs = env.step(action)
        step_count += 1
        
        if obs.done:
            break
            
    final_score = env.grade()
    return final_score

if __name__ == "__main__":
    POLICIES = ["do_nothing", "random", "smart_heuristic", "super_smart_heuristic"]
    
    print(f"{'Task':<22} | {'Do Nothing':<12} | {'Random':<12} | {'Smart':<12} | {'Super Smart Oracle'}")
    print("-" * 80)
    
    for task in TASKS:
        scores = {}
        for policy in POLICIES:
            try:
                score = run_policy(task, policy)
            except Exception as e:
                score = 0.0
            scores[policy] = score
            
        print(f"{task:<22} | {scores['do_nothing']:<12.3f} | {scores['random']:<12.3f} | {scores['smart_heuristic']:<12.3f} | {scores['super_smart_heuristic']:.3f}")

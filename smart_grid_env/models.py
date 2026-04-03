from pydantic import BaseModel
from typing import List, Dict, Optional

class Observation(BaseModel):
    # What the agent SEES each step
    hour: int                          # 0-23, time of day
    grid_frequency: float              # Normal = 50.0 Hz (India standard)
    total_demand_mw: float             # How much power the city needs right now
    available_supply_mw: float         # How much power is available
    solar_output_mw: float             # Solar power right now (0 at night)
    wind_output_mw: float              # Wind power right now
    load_curtailment_mw: float         # How much we've reduced so far
    loads: List[Dict]                  # List of buildings/factories we can control
    # Each load looks like:
    # {"id": "factory_1", "current_mw": 50.0, "reducible_mw": 20.0, "discomfort": 0.3}

class Action(BaseModel):
    # What the agent DOES each step
    curtailments: Dict[str, float]
    # Example: {"factory_1": 10.0, "mall_2": 5.0}
    # Keys = load IDs, Values = how many MW to reduce

class StepResult(BaseModel):
    # What the environment returns after each action
    observation: Observation
    reward: float           # Score for this step (-inf to +1)
    done: bool              # Is the episode over?
    info: Dict              # Extra debug info (cost, blackout_risk, etc.)

from .simulator import GridSimulator
from models import Observation, Action, StepResult
from .tasks import EasyTask, MediumTask, HardTask

class SmartGridEnv:
    def __init__(self, task_name="peak_survival", seed=42):
        self.simulator = GridSimulator(seed=seed)
        self.tasks = {
            "peak_survival": EasyTask(),
            "daily_balance": MediumTask(),
            "cost_minimization_fair": HardTask(),
        }
        self.current_task = self.tasks[task_name]
        self.episode_history = []
        self.hour = 0
        self.done = False

    def reset(self) -> Observation:
        """Start fresh. Called before every new episode."""
        self.hour = 17  # Start at 5pm (before evening peak)
        self.episode_history = []
        self.done = False
        return self._make_observation(curtailment_mw=0)

    def step(self, action: Action) -> StepResult:
        """
        Execute one action (15-minute timestep).
        Returns what the agent observes next + reward.
        """
        demand = self.simulator.get_demand(self.hour)
        solar, wind = self.simulator.get_renewable_output(self.hour)

        # Apply curtailments from agent's action
        total_curtailed = 0
        avg_discomfort = 0
        curtailment_cost = 0

        for load in self.simulator.loads:
            reduction = action.curtailments.get(load["id"], 0)
            max_reducible = load["base_mw"] * load["reducible_fraction"]
            # Clamp: can't reduce more than what's reducible
            actual_reduction = min(reduction, max_reducible)
            total_curtailed += actual_reduction
            avg_discomfort += load["discomfort"] * (actual_reduction / max(1, max_reducible))
            curtailment_cost += actual_reduction * load["discomfort"] * 10  # Cost model

        avg_discomfort = avg_discomfort / len(self.simulator.loads)

        # Effective supply = thermal baseline + renewables - demand + curtailment
        thermal_supply = 150  # Base thermal power plants (always on)
        effective_supply = thermal_supply + solar + wind
        net = effective_supply + total_curtailed - demand  # Surplus if positive

        freq = self.simulator.get_grid_frequency(demand, effective_supply + total_curtailed)

        # Reward: good frequency = +1, blackout = -10, discomfort costs points
        if freq < 49.0:
            reward = -10.0  # Blackout!
        elif freq < 49.5:
            reward = -2.0   # Danger zone
        elif freq < 49.8:
            reward = 0.3    # Acceptable
        else:
            reward = 1.0    # Excellent

        # Penalize discomfort and cost
        reward -= avg_discomfort * 0.5
        reward -= curtailment_cost * 0.001

        # Bonus for using renewables efficiently (not curtailing when solar is ample)
        if solar > 40 and total_curtailed > 50:
            reward -= 0.5   # Penalize: didn't need to curtail this much!

        step_record = {
            "grid_frequency": freq,
            "avg_discomfort": avg_discomfort,
            "curtailment_cost": curtailment_cost,
            "curtailments": action.curtailments,
        }
        self.episode_history.append(step_record)

        self.hour = (self.hour + 1) % 24
        # End episode after 12 steps (3 hours) for easy task
        self.done = len(self.episode_history) >= 12

        obs = self._make_observation(curtailment_mw=total_curtailed)
        return StepResult(observation=obs, reward=reward, done=self.done, info=step_record)

    def state(self) -> dict:
        return {
            "hour": self.hour,
            "episode_length": len(self.episode_history),
            "done": self.done,
            "task": self.current_task.name,
        }

    def grade(self) -> float:
        return self.current_task.grade(self.episode_history)

    def _make_observation(self, curtailment_mw) -> Observation:
        demand = self.simulator.get_demand(self.hour)
        solar, wind = self.simulator.get_renewable_output(self.hour)
        thermal = 150
        supply = thermal + solar + wind
        freq = self.simulator.get_grid_frequency(demand, supply + curtailment_mw)

        loads_view = [
            {
                "id": l["id"],
                "current_mw": l["base_mw"],
                "reducible_mw": l["base_mw"] * l["reducible_fraction"],
                "discomfort": l["discomfort"]
            }
            for l in self.simulator.loads
        ]

        return Observation(
            hour=self.hour,
            grid_frequency=freq,
            total_demand_mw=demand,
            available_supply_mw=supply,
            solar_output_mw=solar,
            wind_output_mw=wind,
            load_curtailment_mw=curtailment_mw,
            loads=loads_view,
        )

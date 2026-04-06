"""
Smart Grid Environment
=======================
The main OpenEnv-compatible environment class. 
Connects the simulator, tasks, and models.
"""

from __future__ import annotations
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Observation, Action, StepResult
from .simulator import GridSimulator
from .tasks import get_task, TASK_REGISTRY


class SmartGridEnv(Environment):
    """
    OpenEnv-compatible Smart Grid Demand Response environment.
    Designed specifically for LLM agents with natural language situation reports.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.simulator = GridSimulator()
        self.current_task = None
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.episode_history = []
        self.hour = 0
        self.day = 1
        self.done = False
        self.total_cost = 0.0
        self.total_discomfort = 0.0
        self.blackout = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "peak_survival",
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode."""
        self.current_task = get_task(task_name)
        
        # Initialize time based on task
        self.hour = self.current_task.start_hour
        self.day = self.current_task.start_day
        
        # Reset simulator
        self.simulator.reset(seed=seed, start_day=self.day)
        if self.current_task.forced_weather:
            self.simulator.weather = self.current_task.forced_weather
            
        # Reset tracking
        self.episode_history = []
        self.done = False
        self.total_cost = 0.0
        self.total_discomfort = 0.0
        self.blackout = False
        
        # Update state
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )

        return self._make_observation(curtailment_mw=0, step_info={})

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> StepResult:
        """Execute one demand response action (typically 1 hour or 15 mins)."""
        if self.done:
            # Re-return last observation if called after done
            obs = self._make_observation(0, {})
            return StepResult(observation=obs, reward=0.0, done=True)

        self._state.step_count += 1
        
        # 1. Advance weather/environment dynamics
        if self._state.step_count % 4 == 0:  # Change weather every few steps or hours
            self.simulator.advance_weather()

        # 2. Get current grid state before action
        demand = self.simulator.get_demand(self.hour, self.day)
        solar, wind = self.simulator.get_renewable_output(self.hour, self.day)
        thermal = self.simulator.THERMAL_BASE_MW
        base_supply = thermal + solar + wind

        # 3. Apply agent's curtailment actions
        curtail_results = self.simulator.apply_curtailments(action.curtailments)
        total_curtailed = curtail_results["total_curtailed_mw"]
        
        # 4. Calculate grid physics (frequency)
        effective_supply = base_supply + total_curtailed
        frequency = self.simulator.get_grid_frequency(demand, effective_supply)
        alert = self.simulator.get_alert_level(frequency)

        # 5. Check for blackout
        if alert == "blackout" or frequency < 49.0:
            self.blackout = True
            self.done = True

        # 6. Update metrics
        self.total_cost += curtail_results["total_cost_inr"]
        self.total_discomfort += curtail_results["avg_discomfort"]
        
        step_info = {
            "grid_frequency_hz": frequency,
            "alert_level": alert,
            "total_demand_mw": demand,
            "total_supply_mw": base_supply,
            "solar_mw": solar,
            "wind_mw": wind,
            "thermal_mw": thermal,
            "total_curtailed_mw": total_curtailed,
            "curtailment_cost_inr": curtail_results["total_cost_inr"],
            "avg_discomfort": curtail_results["avg_discomfort"],
            "per_load_curtailments": curtail_results["per_load"],
            "renewable_pct": (solar + wind) / max(1, base_supply) * 100,
            "hour": self.hour,
            "day": self.day,
            "weather": self.simulator.weather,
            "temperature": self.simulator.temperature,
        }
        
        self.episode_history.append(step_info)

        # 7. Advance time
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day % 365) + 1

        # 8. Check if task objective met or time limit reached
        if self._state.step_count >= self.current_task.episode_steps:
            self.done = True

        obs = self._make_observation(total_curtailed, step_info)
        
        # 9. Simple reward fallback if step_reward isn't defined by task (grading is the primary metric)
        # We use a negative weighted sum of frequency deviation, cost, and discomfort
        freq_penalty = abs(50.0 - frequency) * 5.0
        reward = -(freq_penalty + (curtail_results["total_cost_inr"] / 5000.0) + (curtail_results["avg_discomfort"] * 2.0))
        if self.blackout:
            reward -= 50.0

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info=step_info
        )

    def state(self) -> dict:
        """Return full internal environment state."""
        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "hour": self.hour,
            "day": self.day,
            "total_cost": self.total_cost,
            "total_discomfort": self.total_discomfort,
            "blackout": self.blackout,
            "task": self.current_task.name if self.current_task else None,
            "weather": self.simulator.weather,
            "temperature": self.simulator.temperature,
        }

    def grade(self) -> float:
        """Grade the episode using the current task's grader."""
        if not self.current_task:
            return 0.0
        return self.current_task.grade(self.episode_history)

    def _make_observation(self, curtailment_mw: float, step_info: dict) -> Observation:
        """Create a rich Observation including the situation report."""
        demand = self.simulator.get_demand(self.hour, self.day)
        solar, wind = self.simulator.get_renewable_output(self.hour, self.day)
        thermal = self.simulator.THERMAL_BASE_MW
        supply = thermal + solar + wind
        
        # Current (pre-action for next step) frequency
        freq = self.simulator.get_grid_frequency(demand, supply)
        alert = self.simulator.get_alert_level(freq)
        
        report = self._generate_situation_report(freq, alert, demand, supply, solar, wind)

        return Observation(
            hour=self.hour,
            day=self.day,
            step_number=self._state.step_count,
            grid_frequency_hz=freq,
            alert_level=alert,
            total_demand_mw=round(demand, 2),
            total_supply_mw=round(supply, 2),
            solar_output_mw=round(solar, 2),
            wind_output_mw=round(wind, 2),
            thermal_output_mw=round(thermal, 2),
            supply_deficit_mw=round(max(0, demand - supply), 2),
            weather=self.simulator.weather,
            temperature_c=round(self.simulator.temperature, 1),
            loads=self.simulator.get_load_views(),
            total_curtailed_mw=curtailment_mw,
            curtailment_cost_inr=step_info.get("curtailment_cost_inr", 0.0),
            renewable_utilization_pct=round((solar + wind) / max(1, supply) * 100, 1),
            cumulative_cost_inr=self.total_cost,
            cumulative_discomfort=self.total_discomfort,
            blackout_occurred=self.blackout,
            situation_report=report
        )

    def _generate_situation_report(self, freq: float, alert: str, demand: float, supply: float, solar: float, wind: float) -> str:
        """Generate a human-readable briefing for LLM reasoning."""
        lines = []
        
        # 1. Header & Urgency
        if alert == "blackout":
            lines.append("🚨 SYSTEM CARNAGE: GRID BLACKOUT DETECTED. ALL SERVICE INTERRUPTED.")
        elif alert == "critical":
            lines.append(f"⚠️ CRITICAL EMERGENCY: Grid frequency at {freq:.2f}Hz. Automatic load shedding imminent.")
        elif alert == "warning":
            lines.append(f"🔶 WARNING: Grid stability degrading. Frequency at {freq:.2f}Hz. Action required.")
        else:
            lines.append(f"✅ NORMAL: Grid stable at {freq:.2f}Hz.")

        # 2. Situational Context
        time_str = f"{self.hour:02d}:00"
        weather_str = self.simulator.weather.upper()
        lines.append(f"Time: {time_str} | Weather: {weather_str} ({self.simulator.temperature:.1f}°C)")
        
        deficit = demand - supply
        if deficit > 0:
            lines.append(f"Deficit: {deficit:.1f}MW shortfall. Demand ({demand:.1f}MW) exceeds Supply ({supply:.1f}MW).")
        else:
            lines.append(f"Surplus: {abs(deficit):.1f}MW excess. Supply ({supply:.1f}MW) covers Demand ({demand:.1f}MW).")

        # 3. Component Updates
        lines.append(f"Supply Breakdown: Thermal={self.simulator.THERMAL_BASE_MW:.1f}MW | Solar={solar:.1f}MW | Wind={wind:.1f}MW")
        
        if self.hour >= 17 and self.hour <= 22:
            lines.append("⚡ NOTE: Currently in evening peak demand hours. Solar declining fast.")
        if self.simulator.weather == "heatwave":
            lines.append("🔥 NOTE: Heatwave active. AC load is exceptionally high and taxing the system.")
        if self.simulator.weather == "storm":
            lines.append("⛈️ NOTE: Storm conditions. Solar output nearly zero; wind erratic.")

        # 4. Action Recommendation Hint (without giving the answer)
        critical_loads = [l for l in self.simulator.loads if l["priority"] == "critical"]
        lines.append(f"Priority Check: {len(critical_loads)} critical infrastructure loads active. MUST PROTECT HOSPITAL AND METRO.")

        return "\n".join(lines)

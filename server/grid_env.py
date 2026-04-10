"""
Smart Grid Environment (Phase 2)
==================================
The main OpenEnv-compatible environment class.
Phase 2: battery integration, cascading failures, enhanced situation reports,
         richer info metadata, and 5-task support.
"""

from __future__ import annotations
import math
import uuid
from collections import deque
from typing import Any, Optional, Deque

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Observation, Action
from .simulator import GridSimulator
from .tasks import get_task, TASK_REGISTRY


# ── Gini coefficient helper ──────────────────────────────────────────────────

def gini(values):
    """Compute Gini coefficient of a list (0=perfectly equal, 1=maximally unequal)."""
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    sorted_v = sorted(values)
    cumsum = 0.0
    for i, v in enumerate(sorted_v):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(sorted_v))


class SmartGridEnv(Environment):
    """
    OpenEnv-compatible Smart Grid Demand Response environment.
    Designed for LLM agents with natural language situation reports,
    cascading failure mechanics, and battery energy storage.
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
        self.cascade_events = 0
        self._freq_history: Deque[float] = deque(maxlen=5)
        self._renewable_mwh_used = 0.0
        self._total_battery_throughput_mwh = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "peak_survival",
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment for a new episode."""
        self.current_task = get_task(task_name)

        self.hour = self.current_task.start_hour
        self.day = self.current_task.start_day

        # Reset simulator with task-specific options
        self.simulator = GridSimulator(
            seed=seed or 42,
            thermal_multiplier=getattr(self.current_task, "thermal_multiplier", 1.0),
        )
        self.simulator.reset(
            seed=seed,
            start_day=self.day,
            forced_weather=self.current_task.forced_weather,
        )

        # Reset tracking
        self.episode_history = []
        self.done = False
        self.total_cost = 0.0
        self.total_discomfort = 0.0
        self.blackout = False
        self.cascade_events = 0
        self._freq_history = deque(maxlen=5)
        self._renewable_mwh_used = 0.0
        self._total_battery_throughput_mwh = 0.0

        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )

        return self._make_observation(curtailment_mw=0.0, step_info={})

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one demand response action (1 hour)."""
        if self.done:
            return self._make_observation(0.0, {}, reward=0.0, done=True)

        self._state.step_count += 1

        # 1. Advance weather every 4 steps (~every 4 hours)
        if self._state.step_count % 4 == 0:
            self.simulator.advance_weather()

        # 2. Current grid state (pre-action)
        demand = self.simulator.get_demand(self.hour, self.day)
        solar, wind = self.simulator.get_renewable_output(self.hour, self.day)
        thermal = self.simulator.thermal_mw
        base_supply = thermal + solar + wind

        # 3. Apply battery action (affects effective supply/demand)
        battery_result = self.simulator.apply_battery_action(
            action.battery_action, action.battery_mw
        )
        # battery net_supply_mw: positive = supply added (discharge), negative = demand added (charge)
        battery_net = battery_result["net_supply_mw"]

        # Track battery throughput for grading renewable_transition task
        self._total_battery_throughput_mwh += battery_result["actual_mw"]

        # 4. Apply curtailments
        curtail_results = self.simulator.apply_curtailments(
            action.curtailments, hour=self.hour
        )
        total_curtailed = curtail_results["total_curtailed_mw"]

        # 5. Effective supply = base + curtailment savings + battery discharge
        effective_supply = base_supply + total_curtailed + battery_net

        # 6. Calculate frequency
        frequency = self.simulator.get_grid_frequency(demand, effective_supply)
        alert = self.simulator.get_alert_level(frequency)
        self._freq_history.append(frequency)

        # 7. Process cascading failures
        newly_tripped = self.simulator.process_cascading_failures(frequency)
        if newly_tripped:
            self.cascade_events += len(newly_tripped)

        # 8. Check for blackout (full system collapse)
        if alert == "blackout" or frequency < self.simulator.FREQ_BLACKOUT:
            self.blackout = True
            self.done = True

        # 9. Update cumulative metrics
        self.total_cost += curtail_results["total_cost_inr"]
        self.total_discomfort += curtail_results["avg_discomfort"]
        self._renewable_mwh_used += (solar + wind)

        # 10. Compute reward (continuous 0–1 signal)
        reward = self._compute_reward(
            frequency, curtail_results, demand, base_supply, solar, wind
        )

        step_info = {
            "grid_frequency_hz": frequency,
            "alert_level": alert,
            "total_demand_mw": demand,
            "total_supply_mw": base_supply,
            "effective_supply_mw": effective_supply,
            "solar_mw": solar,
            "wind_mw": wind,
            "thermal_mw": thermal,
            "total_curtailed_mw": total_curtailed,
            "curtailment_cost_inr": curtail_results["total_cost_inr"],
            "avg_discomfort": curtail_results["avg_discomfort"],
            "per_load_curtailments": curtail_results["per_load"],
            "renewable_pct": (solar + wind) / max(1.0, base_supply) * 100,
            "battery_result": battery_result,
            "battery_soc_pct": self.simulator.battery.soc_pct,
            "newly_tripped_loads": newly_tripped,
            "tripped_loads": list(self.simulator.tripped_loads),
            "tou_rate": curtail_results["tou_rate"],
            "hour": self.hour,
            "day": self.day,
            "weather": self.simulator.weather,
            "temperature": self.simulator.temperature,
        }

        self.episode_history.append(step_info)

        # 11. Advance time
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day % 365) + 1

        # 12. Check episode completion
        if self._state.step_count >= self.current_task.episode_steps:
            self.done = True

        return self._make_observation(total_curtailed, step_info, reward=reward, done=self.done)

    def _compute_reward(
        self, frequency: float, curtail_results: dict,
        demand: float, supply: float, solar: float, wind: float
    ) -> float:
        """
        Continuous reward signal in [0, 1] range.
        Combines frequency stability, cost efficiency, and discomfort—
        all normalized so the signal is dense and meaningful.
        """
        # 1. Frequency component (0–1): penalises deviation from 50Hz
        freq_dev = abs(50.0 - frequency)
        freq_score = max(0.0, 1.0 - freq_dev / 2.5)  # 0 at ±2.5Hz, 1 at nominal

        # 2. Cost component: normalised against a ~reasonable per-step cost
        step_cost = curtail_results["total_cost_inr"]
        cost_score = max(0.0, 1.0 - step_cost / 100_000)

        # 3. Discomfort component
        discomfort_score = max(0.0, 1.0 - curtail_results["avg_discomfort"])

        # Weighted sum
        reward = 0.5 * freq_score + 0.3 * cost_score + 0.2 * discomfort_score

        # Heavy penalty for blackout
        if self.blackout:
            reward = 0.0

        return round(reward, 4)

    @property
    def state(self) -> State:
        return self._state

    def state_dict(self) -> dict:
        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "hour": self.hour,
            "day": self.day,
            "total_cost": self.total_cost,
            "total_discomfort": self.total_discomfort,
            "blackout": self.blackout,
            "cascade_events": self.cascade_events,
            "task": self.current_task.name if self.current_task else None,
            "weather": self.simulator.weather,
            "temperature": self.simulator.temperature,
            "battery_soc_pct": self.simulator.battery.soc_pct,
            "tripped_loads": list(self.simulator.tripped_loads),
        }

    def grade(self) -> float:
        """Grade the episode using the current task's grader."""
        if not self.current_task:
            return 0.0
        return self.current_task.grade(self.episode_history)

    # ── Observation builder ──────────────────────────────────────────────────

    def _make_observation(
        self, curtailment_mw: float, step_info: dict,
        reward: float = 0.0, done: bool = False
    ) -> Observation:
        """Create a rich Observation including the situation report."""
        demand = self.simulator.get_demand(self.hour, self.day)
        solar, wind = self.simulator.get_renewable_output(self.hour, self.day)
        thermal = self.simulator.thermal_mw
        supply = thermal + solar + wind

        freq = self.simulator.get_grid_frequency(demand, supply)
        alert = self.simulator.get_alert_level(freq)

        # Richer info dict
        curtailment_amounts = {
            lid: step_info.get("per_load_curtailments", {}).get(lid, {}).get("actual_mw", 0.0)
            for lid in [l["id"] for l in self.simulator.loads]
        }
        fairness = gini(list(curtailment_amounts.values()))

        # Projected next-step deficit
        next_demand_est = self.simulator.get_demand((self.hour + 1) % 24, self.day)
        next_solar_est, next_wind_est = self.simulator.get_renewable_output((self.hour + 1) % 24, self.day)
        next_supply_est = thermal + next_solar_est + next_wind_est
        projected_deficit = max(0.0, next_demand_est - next_supply_est)

        info = {
            **step_info,
            "frequency_history": list(self._freq_history),
            "tripped_loads": list(self.simulator.tripped_loads),
            "battery_soc_pct": self.simulator.battery.soc_pct,
            "fairness_gini": round(fairness, 4),
            "projected_deficit_mw": round(projected_deficit, 1),
            "cascade_events_total": self.cascade_events,
            "low_freq_consecutive_steps": self.simulator.low_freq_steps,
        }

        report = self._generate_situation_report(
            freq, alert, demand, supply, solar, wind, projected_deficit
        )

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
            supply_deficit_mw=round(max(0.0, demand - supply), 2),
            battery_soc_pct=self.simulator.battery.soc_pct,
            battery_capacity_mwh=self.simulator.battery.CAPACITY_MWH,
            battery_hours_remaining=self.simulator.battery.hours_of_discharge_remaining(),
            tripped_loads=list(self.simulator.tripped_loads),
            low_freq_consecutive_steps=self.simulator.low_freq_steps,
            weather=self.simulator.weather,
            temperature_c=round(self.simulator.temperature, 1),
            loads=self.simulator.get_load_views(hour=self.hour, day=self.day),
            total_curtailed_mw=curtailment_mw,
            curtailment_cost_inr=step_info.get("curtailment_cost_inr", 0.0),
            renewable_utilization_pct=round((solar + wind) / max(1.0, supply) * 100, 1),
            tou_rate_inr_per_kwh=step_info.get("tou_rate", 8.0),
            cumulative_cost_inr=round(self.total_cost, 2),
            cumulative_discomfort=round(self.total_discomfort, 4),
            blackout_occurred=self.blackout,
            cascade_events=self.cascade_events,
            situation_report=report,
            reward=reward,
            done=done,
            info=info,
        )

    # ── Situation Report ─────────────────────────────────────────────────────

    def _generate_situation_report(
        self, freq: float, alert: str, demand: float, supply: float,
        solar: float, wind: float, projected_deficit: float
    ) -> str:
        """Generate a rich, strategic situation briefing for LLM reasoning."""
        lines = []
        thermal = self.simulator.thermal_mw
        battery = self.simulator.battery
        balance = supply - demand
        hour = self.hour

        # 1. Header with urgency
        if alert == "blackout":
            lines.append("🚨 SYSTEM BLACKOUT: Grid has collapsed. All service interrupted.")
        elif alert == "emergency":
            lines.append(f"🆘 EMERGENCY: Grid frequency at {freq:.2f}Hz — AUTO-DISCONNECT ACTIVE. Loads tripping NOW.")
        elif alert == "critical":
            lines.append(f"⚠️  CRITICAL: Frequency at {freq:.2f}Hz. Auto-disconnect in {max(0, self.simulator.CRITICAL_GRACE_STEPS - self.simulator.low_freq_steps)} more steps if unresolved.")
        elif alert == "warning":
            lines.append(f"🔶 WARNING: Frequency degrading at {freq:.2f}Hz. Intervention needed soon.")
        else:
            lines.append(f"✅ NORMAL: Grid stable at {freq:.2f}Hz.")

        # 2. Time, weather, season context
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_name = month_names[(self.day // 30) % 12]
        season_note = {
            "Jan": "Cool Winter", "Feb": "Late Winter", "Mar": "Spring",
            "Apr": "Hot Spring", "May": "Pre-Monsoon Heat", "Jun": "Peak Summer",
            "Jul": "Monsoon", "Aug": "Monsoon", "Sep": "Late Monsoon",
            "Oct": "Post-Monsoon", "Nov": "Mild Autumn", "Dec": "Early Winter"
        }.get(month_name, "")
        weather_desc = self.simulator.weather_engine.description()
        lines.append(
            f"Time: {hour:02d}:00 | Weather: {self.simulator.weather.upper()} — {weather_desc} ({self.simulator.temperature:.1f}°C)"
            f" | Day {self.day} ({month_name} — {season_note})"
        )

        # 3. Supply vs Demand block
        solar_trend = "↘ declining" if 14 <= hour <= 18 else ("↗ rising" if 6 <= hour <= 12 else "—")
        demand_trend = "↗ RISING" if 16 <= hour <= 21 else ("↘ falling" if 22 <= hour or hour <= 5 else "→ stable")
        lines.append("")
        lines.append("📊 SUPPLY vs DEMAND:")
        lines.append(f"  Supply:  {supply:.1f}MW  (Thermal: {thermal:.0f}MW | Solar: {solar:.1f}MW {solar_trend} | Wind: {wind:.1f}MW)")
        lines.append(f"  Demand:  {demand:.1f}MW {demand_trend}")
        if balance >= 0:
            lines.append(f"  Balance: +{balance:.1f}MW surplus")
        else:
            lines.append(f"  Balance: {balance:.1f}MW ← DEFICIT — grid frequency will fall!")

        # 4. Battery status
        hours_left = battery.hours_of_discharge_remaining()
        lines.append("")
        
        bat_warn = ""
        if battery.soc_pct >= 90.0:
            bat_warn = " ⚡ (Nearly full — consider discharging)"
        elif battery.soc_pct <= 15.0:
            bat_warn = " ⚠️ (Critically low — conserve or charge)"
            
        lines.append(
            f"🔋 BATTERY: {battery.stored_mwh:.1f}/{battery.CAPACITY_MWH:.0f} MWh "
            f"({battery.soc_pct:.0f}% charged){bat_warn} | "
            f"Can discharge up to {battery.MAX_RATE_MW:.0f}MW for {hours_left:.1f}h"
        )

        # 5. Forecast
        hours_to_sunset = max(0, 18 - hour)
        peak_hour = 19
        hours_to_peak = (peak_hour - hour) % 24
        lines.append("")
        lines.append("⚠️  FORECAST:")
        if hours_to_sunset > 0 and solar > 5:
            lines.append(f"  - Solar drops to 0MW at sunset (~{hours_to_sunset}h from now)")
        if hour < peak_hour:
            lines.append(f"  - Evening demand peak ~19:00 ({hours_to_peak}h away) — expect +60-80MW surge")
        if projected_deficit > 0:
            lines.append(f"  - Projected deficit next hour: {projected_deficit:.1f}MW WITHOUT action")
        if self.simulator.weather == "monsoon" and solar < 5:
            lines.append("  - ⛈️ Monsoon: Solar near zero all day. Battery is your only buffer.")
        if self.simulator.weather == "heatwave" and hour >= 14:
            lines.append("  - 🔥 Heatwave: AC load surging. Evening peak will be +35% above normal.")

        # 6. Cascading risk
        lines.append("")
        tripped = self.simulator.tripped_loads
        low_steps = self.simulator.low_freq_steps
        grace_left = max(0, self.simulator.CRITICAL_GRACE_STEPS - low_steps)
        if tripped:
            lines.append(f"⚡ CASCADING: {len(tripped)} load(s) AUTO-DISCONNECTED: {', '.join(tripped)}")
            lines.append(f"   ⚠️ Tripped loads rejoin after cooldown ({self.simulator.RECONNECT_COOLDOWN} steps).")
        else:
            lines.append(f"⚡ CASCADING RISK: 0 loads tripped. Auto-disconnect triggers at <49.2Hz for {self.simulator.CRITICAL_GRACE_STEPS} steps.")
        if low_steps > 0 and freq < self.simulator.FREQ_CRITICAL:
            lines.append(f"   ⚠️ ⏱️ CASCADE COUNTDOWN: Already {low_steps} consecutive step(s) below 49.2Hz. {grace_left} step(s) until cascade!")

        # 7. Load status (sorted by curtailment priority)
        lines.append("")
        lines.append("🏭 LOAD STATUS (sorted by curtailment priority — LOW first):")
        sorted_loads = sorted(
            self.simulator.get_load_views(hour=self.hour, day=self.day),
            key=lambda l: {"low": 0, "medium": 1, "high": 2, "critical": 3}[l["priority"]]
        )
        for lv in sorted_loads:
            trip_str = " 🔴 TRIPPED (offline)" if lv["tripped"] else ""
            crit_str = " ⛔ DO NOT CURTAIL" if lv["priority"] == "critical" else ""
            reconnect = f" (reconnects in {lv['reconnect_in']}h)" if lv["reconnect_in"] > 0 else ""
            lines.append(
                f"  [{lv['priority'].upper()[:3]}] {lv['name']}: {lv['current_mw']:.0f}MW "
                f"(can reduce {lv['reducible_mw']:.0f}MW, curtailed {lv['curtailed_this_episode']}x)"
                f"{trip_str}{reconnect}{crit_str}"
            )

        # 8. Battery action hint
        lines.append("")
        lines.append("🎮 AVAILABLE ACTIONS:")
        lines.append("  - curtailments: {load_id: mw_to_reduce} — reduce load demand")
        lines.append("  - battery_action: 'charge' | 'discharge' | 'idle'  +  battery_mw: 0-25")

        return "\n".join(lines)

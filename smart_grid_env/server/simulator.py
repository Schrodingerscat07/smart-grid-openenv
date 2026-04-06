"""
Grid Simulator — Physics Engine
================================
Models the physical behavior of an Indian city power grid:
- Demand curves (morning/evening peaks, seasonal, weather-driven)
- Renewable generation (solar bell curve, wind patterns)
- Grid frequency dynamics (supply-demand balance)
- Weather system (clear, cloudy, heatwave, storm, monsoon)
- Controllable loads with priority levels
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Tuple, Optional


# ── Load definitions ────────────────────────────────────────────────────────
# Priority levels: "critical" (hospitals) > "high" (residential) > "medium" (commercial) > "low" (industrial)
# discomfort_factor: 0.0 (no one cares) to 1.0 (causes real suffering)

DEFAULT_LOADS = [
    # Industrial — big, flexible, low discomfort
    {"id": "steel_plant",      "name": "Tata Steel Works",        "base_mw": 80,  "reducible_fraction": 0.40, "priority": "low",      "discomfort_factor": 0.3},
    {"id": "cement_factory",   "name": "UltraTech Cement",        "base_mw": 60,  "reducible_fraction": 0.50, "priority": "low",      "discomfort_factor": 0.25},
    {"id": "textile_mill",     "name": "Raymond Textile Mill",    "base_mw": 30,  "reducible_fraction": 0.60, "priority": "low",      "discomfort_factor": 0.35},
    # IT/Commercial — moderate, some flexibility
    {"id": "it_park",          "name": "Infosys IT Campus",       "base_mw": 25,  "reducible_fraction": 0.20, "priority": "medium",   "discomfort_factor": 0.6},
    {"id": "shopping_mall_1",  "name": "Phoenix Mall",            "base_mw": 15,  "reducible_fraction": 0.30, "priority": "medium",   "discomfort_factor": 0.65},
    {"id": "office_complex",   "name": "DLF Cyber Hub Offices",   "base_mw": 20,  "reducible_fraction": 0.25, "priority": "medium",   "discomfort_factor": 0.7},
    # Residential — small individual, low flexibility, high discomfort
    {"id": "residential_north","name": "North Delhi Colonies",    "base_mw": 40,  "reducible_fraction": 0.10, "priority": "high",     "discomfort_factor": 0.9},
    {"id": "residential_south","name": "South Delhi Residential", "base_mw": 35,  "reducible_fraction": 0.10, "priority": "high",     "discomfort_factor": 0.9},
    # Critical infrastructure — almost no flexibility, must not curtail
    {"id": "hospital",         "name": "AIIMS Hospital",          "base_mw": 12,  "reducible_fraction": 0.05, "priority": "critical", "discomfort_factor": 1.0},
    {"id": "metro_rail",       "name": "Delhi Metro Network",     "base_mw": 18,  "reducible_fraction": 0.08, "priority": "critical", "discomfort_factor": 0.95},
]


# ── Weather system ──────────────────────────────────────────────────────────

WEATHER_TRANSITIONS = {
    # current_weather → {next_weather: probability}
    "clear":    {"clear": 0.70, "cloudy": 0.20, "heatwave": 0.08, "storm": 0.02, "monsoon": 0.00},
    "cloudy":   {"clear": 0.30, "cloudy": 0.45, "heatwave": 0.02, "storm": 0.15, "monsoon": 0.08},
    "heatwave": {"clear": 0.15, "cloudy": 0.10, "heatwave": 0.65, "storm": 0.05, "monsoon": 0.05},
    "storm":    {"clear": 0.20, "cloudy": 0.40, "heatwave": 0.00, "storm": 0.30, "monsoon": 0.10},
    "monsoon":  {"clear": 0.05, "cloudy": 0.25, "heatwave": 0.00, "storm": 0.15, "monsoon": 0.55},
}

WEATHER_EFFECTS = {
    #                   solar_mult  wind_mult  demand_mult  temp_offset
    "clear":    {"solar_mult": 1.0,  "wind_mult": 1.0,  "demand_mult": 1.0,  "temp_offset": 0},
    "cloudy":   {"solar_mult": 0.4,  "wind_mult": 1.2,  "demand_mult": 0.95, "temp_offset": -3},
    "heatwave": {"solar_mult": 1.1,  "wind_mult": 0.5,  "demand_mult": 1.35, "temp_offset": 8},
    "storm":    {"solar_mult": 0.1,  "wind_mult": 0.3,  "demand_mult": 0.85, "temp_offset": -5},
    "monsoon":  {"solar_mult": 0.2,  "wind_mult": 0.6,  "demand_mult": 0.90, "temp_offset": -2},
}


class GridSimulator:
    """
    Physics engine for an Indian city power grid.

    Tracks demand, renewable generation, grid frequency, weather,
    and controllable loads across hourly timesteps.
    """

    # Grid constants
    THERMAL_BASE_MW: float = 150.0       # Always-on coal/gas plants
    SOLAR_PEAK_MW: float = 60.0          # Installed solar capacity
    WIND_PEAK_MW: float = 40.0           # Installed wind capacity
    NOMINAL_FREQUENCY: float = 50.0      # India grid standard

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.loads = [load.copy() for load in DEFAULT_LOADS]
        self.weather = "clear"
        self.temperature = 35.0
        self._curtailment_history: Dict[str, int] = {l["id"]: 0 for l in self.loads}

    def reset(self, seed: Optional[int] = None, start_day: int = 1) -> None:
        """Reset simulator state for a new episode."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.loads = [load.copy() for load in DEFAULT_LOADS]
        self.weather = "clear"
        # Temperature based on season (Delhi climate approximation)
        month = (start_day // 30) % 12 + 1
        if month in (4, 5, 6):
            self.temperature = 38.0 + self.rng.gauss(0, 3)
        elif month in (12, 1, 2):
            self.temperature = 15.0 + self.rng.gauss(0, 3)
        elif month in (7, 8, 9):
            self.temperature = 32.0 + self.rng.gauss(0, 2)
            self.weather = self.rng.choice(["cloudy", "monsoon"])
        else:
            self.temperature = 28.0 + self.rng.gauss(0, 2)

        self._curtailment_history = {l["id"]: 0 for l in self.loads}

    def advance_weather(self) -> None:
        """Stochastic weather transition (Markov chain)."""
        transitions = WEATHER_TRANSITIONS[self.weather]
        r = self.rng.random()
        cumulative = 0.0
        for next_weather, prob in transitions.items():
            cumulative += prob
            if r <= cumulative:
                self.weather = next_weather
                break
        # Update temperature with weather
        effects = WEATHER_EFFECTS[self.weather]
        base_temp = self.temperature
        self.temperature = base_temp + effects["temp_offset"] * 0.3 + self.rng.gauss(0, 1)
        self.temperature = max(5.0, min(50.0, self.temperature))

    def get_demand(self, hour: int, day: int) -> float:
        """
        Total city demand in MW. Follows realistic Indian load curve:
        - Low at night (1am-5am)
        - Morning ramp (6am-10am) — offices, factories start
        - Midday plateau (11am-4pm) — AC load if hot
        - Evening peak (5pm-10pm) — THE CRISIS TIME
        """
        effects = WEATHER_EFFECTS[self.weather]

        # Base load (always-on: streetlights, hospitals, water pumps)
        base = 180.0

        # Morning industrial ramp
        if 6 <= hour <= 10:
            morning = 50.0 * math.sin(math.pi * (hour - 6) / 4)
        else:
            morning = 0.0

        # Midday AC load (proportional to temperature)
        if 11 <= hour <= 16:
            ac_factor = max(0, (self.temperature - 28) / 17)  # 0 at 28°C, ~1 at 45°C
            midday = 40.0 * ac_factor
        else:
            midday = 0.0

        # Evening peak (everyone home, cooking, TV, AC still running)
        if 17 <= hour <= 22:
            evening = 80.0 * math.sin(math.pi * (hour - 17) / 5)
        else:
            evening = 0.0

        # Night trough
        if 0 <= hour <= 5:
            night_reduction = -40.0
        else:
            night_reduction = 0.0

        # Seasonal modifier (summer = higher demand)
        seasonal = 1.0 + 0.15 * math.sin(2 * math.pi * (day - 120) / 365)

        # Weather-driven demand multiplier
        total = (base + morning + midday + evening + night_reduction) * seasonal * effects["demand_mult"]

        # Stochastic noise (±3%)
        noise = self.rng.gauss(0, total * 0.03)

        return max(100.0, total + noise)

    def get_renewable_output(self, hour: int, day: int) -> Tuple[float, float]:
        """Returns (solar_mw, wind_mw) after weather effects."""
        effects = WEATHER_EFFECTS[self.weather]

        # Solar: bell curve 6am-6pm, peak at noon
        if 6 <= hour <= 18:
            solar_base = self.SOLAR_PEAK_MW * math.sin(math.pi * (hour - 6) / 12)
            # Seasonal: more sun in summer
            seasonal_solar = 1.0 + 0.2 * math.sin(2 * math.pi * (day - 80) / 365)
            solar = solar_base * seasonal_solar * effects["solar_mult"]
            solar += self.rng.gauss(0, 3)  # Cloud variance
        else:
            solar = 0.0

        # Wind: higher at night and in winter, very variable
        wind_base = 15.0 + 10.0 * math.cos(2 * math.pi * hour / 24)  # Night bias
        seasonal_wind = 1.0 + 0.3 * math.cos(2 * math.pi * (day - 1) / 365)  # Winter bias
        wind = wind_base * seasonal_wind * effects["wind_mult"]
        wind += self.rng.gauss(0, 5)  # Wind is noisy

        return max(0.0, solar), max(0.0, wind)

    def get_grid_frequency(self, demand: float, effective_supply: float) -> float:
        """
        Grid frequency based on supply-demand imbalance.
        India standard: 50.0 Hz.
        - Oversupply → frequency rises
        - Undersupply → frequency drops
        - Below 49.0 Hz → cascading blackout risk
        """
        imbalance = effective_supply - demand
        # Each 100MW imbalance shifts frequency by ~0.25Hz (simplified model)
        freq_shift = (imbalance / 100.0) * 0.25
        frequency = self.NOMINAL_FREQUENCY + freq_shift
        # Add small noise
        frequency += self.rng.gauss(0, 0.02)
        return round(max(47.0, min(52.0, frequency)), 3)

    def get_alert_level(self, frequency: float) -> str:
        """Map frequency to alert level."""
        if frequency >= 49.8:
            return "normal"
        elif frequency >= 49.5:
            return "warning"
        elif frequency >= 49.0:
            return "critical"
        else:
            return "blackout"

    def get_load_views(self) -> List[Dict]:
        """Build the observation-friendly view of controllable loads."""
        views = []
        for load in self.loads:
            views.append({
                "id": load["id"],
                "name": load["name"],
                "current_mw": load["base_mw"],
                "reducible_mw": round(load["base_mw"] * load["reducible_fraction"], 1),
                "priority": load["priority"],
                "discomfort_factor": load["discomfort_factor"],
                "curtailed_this_episode": self._curtailment_history.get(load["id"], 0),
            })
        return views

    def apply_curtailments(self, curtailments: Dict[str, float]) -> Dict:
        """
        Apply agent's curtailment decisions.
        Returns summary: {total_curtailed, total_cost, avg_discomfort, per_load_details}
        """
        total_curtailed = 0.0
        total_cost = 0.0
        total_discomfort = 0.0
        per_load = {}

        for load in self.loads:
            reduction = curtailments.get(load["id"], 0.0)
            if reduction <= 0:
                continue

            max_reducible = load["base_mw"] * load["reducible_fraction"]
            actual = min(reduction, max_reducible)
            actual = max(0.0, actual)

            # Cost model: ₹ per MW curtailed, weighted by discomfort
            # Industrial loads are cheaper to curtail than residential
            base_rate_inr = 8000.0  # ₹8000 per MW-hour base
            cost = actual * base_rate_inr * (0.5 + load["discomfort_factor"])
            discomfort = load["discomfort_factor"] * (actual / max(1.0, max_reducible))

            # Priority penalty: curtailing critical loads is extremely costly
            if load["priority"] == "critical":
                cost *= 5.0
                discomfort *= 3.0
            elif load["priority"] == "high":
                cost *= 2.0
                discomfort *= 1.5

            total_curtailed += actual
            total_cost += cost
            total_discomfort += discomfort
            self._curtailment_history[load["id"]] = self._curtailment_history.get(load["id"], 0) + 1

            per_load[load["id"]] = {
                "requested_mw": reduction,
                "actual_mw": round(actual, 2),
                "cost_inr": round(cost, 2),
                "discomfort": round(discomfort, 3),
            }

        avg_discomfort = total_discomfort / max(1, len(per_load)) if per_load else 0.0

        return {
            "total_curtailed_mw": round(total_curtailed, 2),
            "total_cost_inr": round(total_cost, 2),
            "avg_discomfort": round(avg_discomfort, 4),
            "per_load": per_load,
        }

    @property
    def curtailment_history(self) -> Dict[str, int]:
        return self._curtailment_history.copy()

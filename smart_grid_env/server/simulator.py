"""
Grid Simulator — Physics Engine (Phase 2)
==========================================
Models the physical behavior of an Indian city power grid:
- Dynamic per-load demand (time-of-day, temperature-dependent)
- Battery Energy Storage System (50MWh BESS)
- Cascading failure mechanics (frequency thresholds → auto-disconnect)
- Markov weather system with demand/supply effects
- India-specific pricing (₹/kWh time-of-use tariffs)
- Grid frequency dynamics (supply-demand balance)
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Tuple, Optional

from .weather import WeatherEngine, WEATHER_EFFECTS


# ── Load definitions ────────────────────────────────────────────────────────
# Priority: "critical" > "high" > "medium" > "low"
# discomfort_factor: 0.0 (industry, doesn't care) → 1.0 (hospital, catastrophic)

DEFAULT_LOADS = [
    # Industrial — large, flexible, low human discomfort
    {"id": "steel_plant",       "name": "Tata Steel Works",        "base_mw": 80,  "reducible_fraction": 0.40, "priority": "low",      "discomfort_factor": 0.3},
    {"id": "cement_factory",    "name": "UltraTech Cement",        "base_mw": 60,  "reducible_fraction": 0.50, "priority": "low",      "discomfort_factor": 0.25},
    {"id": "textile_mill",      "name": "Raymond Textile Mill",    "base_mw": 30,  "reducible_fraction": 0.60, "priority": "low",      "discomfort_factor": 0.35},
    # IT/Commercial — moderate flexibility
    {"id": "it_park",           "name": "Infosys IT Campus",       "base_mw": 25,  "reducible_fraction": 0.20, "priority": "medium",   "discomfort_factor": 0.6},
    {"id": "shopping_mall_1",   "name": "Phoenix Mall",            "base_mw": 15,  "reducible_fraction": 0.30, "priority": "medium",   "discomfort_factor": 0.65},
    {"id": "office_complex",    "name": "DLF Cyber Hub Offices",   "base_mw": 20,  "reducible_fraction": 0.25, "priority": "medium",   "discomfort_factor": 0.7},
    # Residential — high discomfort, low flexibility
    {"id": "residential_north", "name": "North Delhi Colonies",    "base_mw": 40,  "reducible_fraction": 0.10, "priority": "high",     "discomfort_factor": 0.9},
    {"id": "residential_south", "name": "South Delhi Residential", "base_mw": 35,  "reducible_fraction": 0.10, "priority": "high",     "discomfort_factor": 0.9},
    # Critical infrastructure — essentially non-curtailable
    {"id": "hospital",          "name": "AIIMS Hospital",          "base_mw": 12,  "reducible_fraction": 0.05, "priority": "critical", "discomfort_factor": 1.0},
    {"id": "metro_rail",        "name": "Delhi Metro Network",     "base_mw": 18,  "reducible_fraction": 0.08, "priority": "critical", "discomfort_factor": 0.95},
]

# Priority ordering for auto-disconnect (lowest priority disconnected first)
PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# India time-of-use tariff rates (₹/kWh)
TOU_TARIFFS = {
    "off_peak":   6.0,   # 22:00–06:00
    "normal":     8.0,   # 06:00–17:00
    "peak":      12.5,   # 17:00–22:00
    "super_peak": 16.0,  # 18:00–21:00 (highest)
}


def get_tou_rate(hour: int) -> float:
    """Return the current time-of-use tariff rate in ₹/kWh."""
    if 18 <= hour <= 20:
        return TOU_TARIFFS["super_peak"]
    elif 17 <= hour <= 21:
        return TOU_TARIFFS["peak"]
    elif 6 <= hour <= 16:
        return TOU_TARIFFS["normal"]
    else:
        return TOU_TARIFFS["off_peak"]


# ── Battery Energy Storage System ────────────────────────────────────────────

class BatteryStorage:
    """
    50 MWh Battery Energy Storage System (BESS).
    
    - Capacity: 50 MWh
    - Max charge/discharge rate: 25 MW
    - Round-trip efficiency: 90% (10% losses on each charge-discharge cycle)
    - SOC range: 5%–95% (protect against deep discharge and overcharge)
    """

    CAPACITY_MWH: float = 50.0
    MAX_RATE_MW: float = 25.0
    EFFICIENCY: float = 0.90
    MIN_SOC: float = 0.05   # 5% minimum
    MAX_SOC: float = 0.95   # 95% maximum

    def __init__(self):
        self.soc: float = 0.50  # Start at 50% state of charge

    def reset(self, initial_soc: float = 0.50) -> None:
        self.soc = max(self.MIN_SOC, min(self.MAX_SOC, initial_soc))

    @property
    def stored_mwh(self) -> float:
        return self.soc * self.CAPACITY_MWH

    @property
    def soc_pct(self) -> float:
        return round(self.soc * 100, 1)

    def charge(self, requested_mw: float, dt_hours: float = 1.0) -> float:
        """
        Charge battery with up to requested_mw.
        Returns actual MW drawn from grid (what adds to demand).
        """
        requested_mw = max(0.0, min(requested_mw, self.MAX_RATE_MW))
        energy_in = requested_mw * dt_hours  # MWh
        space_available = (self.MAX_SOC - self.soc) * self.CAPACITY_MWH
        actual_energy = min(energy_in, space_available)
        self.soc += (actual_energy * self.EFFICIENCY) / self.CAPACITY_MWH
        self.soc = min(self.MAX_SOC, self.soc)
        actual_mw = actual_energy / dt_hours
        return round(actual_mw, 2)  # Grid load added

    def discharge(self, requested_mw: float, dt_hours: float = 1.0) -> float:
        """
        Discharge battery to supply requested_mw.
        Returns actual MW injected into grid.
        """
        requested_mw = max(0.0, min(requested_mw, self.MAX_RATE_MW))
        energy_out = requested_mw * dt_hours
        energy_available = (self.soc - self.MIN_SOC) * self.CAPACITY_MWH
        actual_energy = min(energy_out, energy_available)
        self.soc -= actual_energy / (self.CAPACITY_MWH * self.EFFICIENCY)
        self.soc = max(self.MIN_SOC, self.soc)
        actual_mw = actual_energy / dt_hours
        return round(actual_mw, 2)  # Grid supply added

    def hours_of_discharge_remaining(self) -> float:
        """How long can we discharge at max rate?"""
        energy_available = (self.soc - self.MIN_SOC) * self.CAPACITY_MWH
        if self.MAX_RATE_MW <= 0:
            return 0.0
        return round(energy_available / self.MAX_RATE_MW, 1)


# ── Main Simulator ──────────────────────────────────────────────────────────

class GridSimulator:
    """
    Physics engine for an Indian city power grid.
    
    Phase 2 additions:
    - BatteryStorage (BESS)
    - Cascading failure mechanics (tripped_loads, reconnection cooldown)
    - Dynamic per-load demand profiles (bottom-up demand sum)
    - India time-of-use pricing
    """

    THERMAL_BASE_MW: float = 150.0
    SOLAR_PEAK_MW: float = 60.0
    WIND_PEAK_MW: float = 40.0
    NOMINAL_FREQUENCY: float = 50.0

    # Frequency thresholds for cascading failures
    FREQ_WARNING: float = 49.5
    FREQ_CRITICAL: float = 49.2     # 1 load auto-disconnects per step
    FREQ_EMERGENCY: float = 49.0    # 2 loads auto-disconnect per step
    FREQ_BLACKOUT: float = 48.5     # All loads disconnect, episode ends

    # Steps before auto-disconnect triggers (grace period)
    CRITICAL_GRACE_STEPS: int = 2
    # Steps before a tripped load can reconnect
    RECONNECT_COOLDOWN: int = 3

    def __init__(self, seed: int = 42, thermal_multiplier: float = 1.0):
        self.rng = random.Random(seed)
        self.weather_engine = WeatherEngine(seed=seed)
        self.battery = BatteryStorage()
        self.thermal_multiplier = thermal_multiplier  # For renewable_transition task
        self.loads: List[Dict] = []
        self._curtailment_history: Dict[str, int] = {}
        self.tripped_loads: List[str] = []           # Auto-disconnected load IDs
        self._reconnect_cooldown: Dict[str, int] = {}  # load_id → steps until reconnectable
        self.low_freq_steps: int = 0                 # Consecutive steps below FREQ_CRITICAL
        self.temperature: float = 35.0
        self.reset()

    def reset(self, seed: Optional[int] = None, start_day: int = 1, forced_weather: Optional[str] = None) -> None:
        """Reset simulator state for a new episode."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.loads = [load.copy() for load in DEFAULT_LOADS]
        self._curtailment_history = {l["id"]: 0 for l in self.loads}
        self.tripped_loads = []
        self._reconnect_cooldown = {}
        self.low_freq_steps = 0

        # Season-based temperature (Delhi climate)
        month = (start_day // 30) % 12 + 1
        if month in (4, 5, 6):
            self.temperature = 38.0 + self.rng.gauss(0, 3)
        elif month in (12, 1, 2):
            self.temperature = 15.0 + self.rng.gauss(0, 3)
        elif month in (7, 8, 9):
            self.temperature = 32.0 + self.rng.gauss(0, 2)
        else:
            self.temperature = 28.0 + self.rng.gauss(0, 2)
        self.temperature = max(5.0, min(50.0, self.temperature))

        # Set initial weather
        initial_weather = forced_weather or ("monsoon" if month in (7, 8, 9) else "clear")
        self.weather_engine.reset(initial=initial_weather)

        # Battery starts at 50%
        self.battery.reset(initial_soc=0.50)

    # ── Weather ──────────────────────────────────────────────────────────────

    @property
    def weather(self) -> str:
        return self.weather_engine.current

    @weather.setter
    def weather(self, value: str) -> None:
        self.weather_engine.current = value

    def advance_weather(self) -> None:
        """Stochastic weather transition (Markov chain) + temperature update."""
        self.weather_engine.advance()
        effects = self.weather_engine.effects()
        self.temperature += effects["temp_offset"] * 0.3 + self.rng.gauss(0, 1)
        self.temperature = max(5.0, min(50.0, self.temperature))

    # ── Per-load dynamic demand ──────────────────────────────────────────────

    def get_load_demand(self, load: Dict, hour: int, day: int) -> float:
        """
        Return a load's actual current demand in MW based on time, day, temperature.
        Each load has a profile that makes curtailment decisions strategically meaningful.
        """
        load_id = load["id"]
        base = load["base_mw"]
        temp = self.temperature

        if load_id == "steel_plant":
            # 3 shifts: day shift peaks 8am-4pm, night shift 10pm-6am; moderate in between
            if 8 <= hour <= 16:
                factor = 1.0
            elif 22 <= hour or hour <= 6:
                factor = 0.85  # Night shift, slightly reduced
            else:
                factor = 0.75  # Shift change / maintenance
            # Hot weather reduces steel efficiency slightly
            if temp > 40:
                factor *= 0.95
            return base * factor

        elif load_id == "cement_factory":
            # Avoids peak tariff hours; shifts heavy work to off-peak
            if 6 <= hour <= 16:
                factor = 1.0
            elif 17 <= hour <= 21:
                factor = 0.60  # Scaled back during peak tariff
            else:
                factor = 0.90  # Night production
            return base * factor

        elif load_id == "textile_mill":
            # Single day shift + some night maintenance
            if 7 <= hour <= 18:
                factor = 1.0
            elif 19 <= hour <= 22:
                factor = 0.50
            else:
                factor = 0.20  # Minimal lighting/security
            return base * factor

        elif load_id == "it_park":
            # Peak during business hours, background at night
            if 9 <= hour <= 19:
                factor = 1.0
            elif 7 <= hour <= 8 or 20 <= hour <= 22:
                factor = 0.60
            else:
                factor = 0.25  # Servers + cooling only
            # Hot weather = more cooling demand
            if temp > 35:
                factor += 0.10
            return min(base * 1.3, base * factor)

        elif load_id == "shopping_mall_1":
            # Open 10am–10pm, peaks on weekends (approximate: day 0 = Monday)
            is_weekend = (day % 7) in (6, 0)
            if 10 <= hour <= 21:
                factor = 1.3 if is_weekend else 1.0
                # AC load in hot weather
                if temp > 35:
                    factor += 0.15
            elif hour == 9 or hour == 22:
                factor = 0.50
            else:
                factor = 0.10  # Security/maintenance
            return min(base * 1.5, base * factor)

        elif load_id == "office_complex":
            if 9 <= hour <= 18:
                factor = 1.0
            elif 7 <= hour <= 8 or 19 <= hour <= 20:
                factor = 0.50
            else:
                factor = 0.15
            return base * factor

        elif load_id == "residential_north":
            # Evening peak huge: everyone home + cooking + AC
            if 18 <= hour <= 22:
                ac_extra = max(0, (temp - 28) / 20) * 0.50  # up to +50% in extreme heat
                factor = 1.50 + ac_extra
            elif 6 <= hour <= 9:
                factor = 1.20  # Morning routines
            elif 12 <= hour <= 15 and temp > 35:
                factor = 1.30  # Afternoon AC spike
            elif 0 <= hour <= 5:
                factor = 0.45  # Night
            else:
                factor = 0.85
            return base * factor

        elif load_id == "residential_south":
            # Similar to north, slight variation
            if 18 <= hour <= 22:
                ac_extra = max(0, (temp - 28) / 20) * 0.45
                factor = 1.45 + ac_extra
            elif 6 <= hour <= 9:
                factor = 1.15
            elif 0 <= hour <= 5:
                factor = 0.40
            else:
                factor = 0.80
            return base * factor

        elif load_id == "hospital":
            # Constant — critical infrastructure never varies significantly
            return base * (1.0 + self.rng.gauss(0, 0.02))

        elif load_id == "metro_rail":
            # Runs 6am–11pm; off during deep night
            if 6 <= hour <= 22:
                # Peak during rush hours
                if 8 <= hour <= 10 or 17 <= hour <= 20:
                    factor = 1.20
                else:
                    factor = 1.0
            else:
                factor = 0.05  # Depot power only
            return base * factor

        return base  # Fallback

    def get_demand(self, hour: int, day: int) -> float:
        """
        Total city demand: sum of individual load demands (bottom-up).
        Tripped loads don't contribute. Weather effects apply globally.
        """
        effects = WEATHER_EFFECTS[self.weather]
        total = 0.0
        for load in self.loads:
            if load["id"] in self.tripped_loads:
                continue  # Tripped loads are offline
            load_demand = self.get_load_demand(load, hour, day)
            total += load_demand

        # Weather multiplier on total demand
        total *= effects["demand_mult"]
        # Small noise
        total += self.rng.gauss(0, total * 0.02)
        return max(50.0, total)

    # ── Supply ──────────────────────────────────────────────────────────────

    def get_renewable_output(self, hour: int, day: int) -> Tuple[float, float]:
        """Returns (solar_mw, wind_mw) after weather effects."""
        effects = WEATHER_EFFECTS[self.weather]

        # Solar: bell curve 6am–6pm, peak at noon
        if 6 <= hour <= 18:
            solar_base = self.SOLAR_PEAK_MW * math.sin(math.pi * (hour - 6) / 12)
            seasonal_solar = 1.0 + 0.2 * math.sin(2 * math.pi * (day - 80) / 365)
            solar = solar_base * seasonal_solar * effects["solar_mult"]
            solar += self.rng.gauss(0, 3)
        else:
            solar = 0.0

        # Wind: night bias, winter bias
        wind_base = 15.0 + 10.0 * math.cos(2 * math.pi * hour / 24)
        seasonal_wind = 1.0 + 0.3 * math.cos(2 * math.pi * (day - 1) / 365)
        wind = wind_base * seasonal_wind * effects["wind_mult"]
        wind += self.rng.gauss(0, 5)

        return max(0.0, solar), max(0.0, wind)

    @property
    def thermal_mw(self) -> float:
        return self.THERMAL_BASE_MW * self.thermal_multiplier

    # ── Frequency & Cascading Failures ──────────────────────────────────────

    # How much frequency deviates per 100MW of imbalance
    FREQ_SENSITIVITY: float = 0.40

    def get_grid_frequency(self, demand: float, effective_supply: float) -> float:
        """
        Grid frequency based on supply-demand imbalance.
        Each 100MW imbalance ≈ 0.40Hz deviation from nominal 50.0Hz.
        """
        imbalance = effective_supply - demand
        freq_shift = (imbalance / 100.0) * self.FREQ_SENSITIVITY
        frequency = self.NOMINAL_FREQUENCY + freq_shift
        frequency += self.rng.gauss(0, 0.02)
        return round(max(45.0, min(52.0, frequency)), 3)

    def get_alert_level(self, frequency: float) -> str:
        """Map frequency to alert level string."""
        if frequency >= self.FREQ_WARNING:
            return "normal"
        elif frequency >= self.FREQ_CRITICAL:
            return "warning"
        elif frequency >= self.FREQ_EMERGENCY:
            return "critical"
        elif frequency >= self.FREQ_BLACKOUT:
            return "emergency"
        else:
            return "blackout"

    def process_cascading_failures(self, frequency: float) -> List[str]:
        """
        Check frequency and auto-disconnect loads if needed.
        Returns list of newly tripped load IDs this step.
        
        Logic:
        - freq < FREQ_CRITICAL for CRITICAL_GRACE_STEPS consecutive steps: 1 load disconnects
        - freq < FREQ_EMERGENCY: 2 loads disconnect immediately
        - freq < FREQ_BLACKOUT: all remaining loads trip
        """
        newly_tripped = []

        # Decrement reconnection cooldowns
        for lid in list(self._reconnect_cooldown.keys()):
            self._reconnect_cooldown[lid] -= 1
            if self._reconnect_cooldown[lid] <= 0:
                del self._reconnect_cooldown[lid]

        if frequency < self.FREQ_BLACKOUT:
            # Total blackout — trip everything remaining
            for load in self.loads:
                if load["id"] not in self.tripped_loads:
                    self.tripped_loads.append(load["id"])
                    self._reconnect_cooldown[load["id"]] = self.RECONNECT_COOLDOWN
                    newly_tripped.append(load["id"])
            self.low_freq_steps = 0
            return newly_tripped

        if frequency < self.FREQ_EMERGENCY:
            # Emergency: immediately disconnect 2 loads
            self.low_freq_steps = 0
            to_trip = self._get_trip_candidates(count=2)
            for lid in to_trip:
                self.tripped_loads.append(lid)
                self._reconnect_cooldown[lid] = self.RECONNECT_COOLDOWN
                newly_tripped.append(lid)
            return newly_tripped

        if frequency < self.FREQ_CRITICAL:
            self.low_freq_steps += 1
            if self.low_freq_steps >= self.CRITICAL_GRACE_STEPS:
                # Critical for too long: disconnect 1 load
                to_trip = self._get_trip_candidates(count=1)
                for lid in to_trip:
                    self.tripped_loads.append(lid)
                    self._reconnect_cooldown[lid] = self.RECONNECT_COOLDOWN
                    newly_tripped.append(lid)
                self.low_freq_steps = 0
        else:
            # Frequency recovered
            self.low_freq_steps = max(0, self.low_freq_steps - 1)

        return newly_tripped

    def _get_trip_candidates(self, count: int) -> List[str]:
        """Return lowest-priority active loads to auto-disconnect."""
        active = [
            l for l in self.loads
            if l["id"] not in self.tripped_loads
            and l["id"] not in self._reconnect_cooldown
        ]
        # Sort: lowest priority first, then by base_mw descending (bigger demand impact)
        active_sorted = sorted(
            active,
            key=lambda l: (PRIORITY_ORDER[l["priority"]], -l["base_mw"])
        )
        return [l["id"] for l in active_sorted[:count]]

    # ── Curtailments ────────────────────────────────────────────────────────

    def apply_curtailments(self, curtailments: Dict[str, float], hour: int = 12) -> Dict:
        """
        Apply agent's curtailment decisions.
        Uses India time-of-use tariffs for cost calculation.
        Returns summary: {total_curtailed_mw, total_cost_inr, avg_discomfort, per_load}
        """
        tou_rate = get_tou_rate(hour)
        total_curtailed = 0.0
        total_cost = 0.0
        total_discomfort = 0.0
        per_load = {}

        for load in self.loads:
            # Can't curtail tripped loads (they're already offline)
            if load["id"] in self.tripped_loads:
                continue

            reduction = curtailments.get(load["id"], 0.0)
            if reduction <= 0:
                continue

            max_reducible = load["base_mw"] * load["reducible_fraction"]
            actual = max(0.0, min(reduction, max_reducible))

            # Cost: TOU tariff × discomfort multiplier × priority multiplier
            cost_per_mw = tou_rate * 1000 * (0.5 + load["discomfort_factor"])
            cost = actual * cost_per_mw
            discomfort = load["discomfort_factor"] * (actual / max(1.0, max_reducible))

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
                "tou_rate_inr_per_kwh": tou_rate,
            }

        avg_discomfort = total_discomfort / max(1, len(per_load)) if per_load else 0.0

        return {
            "total_curtailed_mw": round(total_curtailed, 2),
            "total_cost_inr": round(total_cost, 2),
            "avg_discomfort": round(avg_discomfort, 4),
            "per_load": per_load,
            "tou_rate": tou_rate,
        }

    # ── Battery actions ──────────────────────────────────────────────────────

    def apply_battery_action(self, battery_action: str, battery_mw: float) -> Dict:
        """
        Apply battery charge/discharge action.
        Returns {net_grid_effect_mw, actual_mw, battery_soc_before, battery_soc_after}
        Where positive net_grid_effect_mw means supply was ADDED (discharge).
        Negative means demand was ADDED (charge).
        """
        soc_before = self.battery.soc_pct
        net_mw = 0.0
        actual_mw = 0.0

        if battery_action == "charge":
            actual_mw = self.battery.charge(abs(battery_mw))
            net_mw = -actual_mw  # Charging ADDS to grid demand (negative for supply)
        elif battery_action == "discharge":
            actual_mw = self.battery.discharge(abs(battery_mw))
            net_mw = actual_mw  # Discharging ADDS to grid supply (positive)

        return {
            "battery_action": battery_action,
            "requested_mw": abs(battery_mw),
            "actual_mw": actual_mw,
            "net_supply_mw": round(net_mw, 2),  # Positive = supply added, negative = demand added
            "battery_soc_before_pct": soc_before,
            "battery_soc_after_pct": self.battery.soc_pct,
        }

    # ── Load views ───────────────────────────────────────────────────────────

    def get_load_views(self, hour: int = 12, day: int = 1) -> List[Dict]:
        """Build observation-friendly view of controllable loads with dynamic demand."""
        views = []
        for load in self.loads:
            current_mw = self.get_load_demand(load, hour, day)
            if load["id"] in self.tripped_loads:
                current_mw = 0.0  # Tripped loads draw no power
            views.append({
                "id": load["id"],
                "name": load["name"],
                "current_mw": round(current_mw, 1),
                "reducible_mw": round(load["base_mw"] * load["reducible_fraction"], 1),
                "priority": load["priority"],
                "discomfort_factor": load["discomfort_factor"],
                "curtailed_this_episode": self._curtailment_history.get(load["id"], 0),
                "tripped": load["id"] in self.tripped_loads,
                "reconnect_in": self._reconnect_cooldown.get(load["id"], 0),
            })
        return views

    @property
    def curtailment_history(self) -> Dict[str, int]:
        return self._curtailment_history.copy()

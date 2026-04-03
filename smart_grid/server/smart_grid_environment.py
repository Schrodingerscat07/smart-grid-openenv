"""
Smart Grid Microgrid RL Environment.

Simulates a small microgrid with:
  • Solar PV array (peak capacity configurable)
  • Wind turbine (peak capacity configurable)
  • Battery Energy Storage System (BESS)
  • Connection to the main utility grid

The RL agent decides each hour how to dispatch renewables, charge/discharge the
battery, and how much energy to buy/sell from/to the main grid.  The goal is to
minimise operating cost while keeping CO2 emissions low and meeting demand
reliably.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SmartGridAction, SmartGridObservation
except ImportError:
    from models import SmartGridAction, SmartGridObservation


# ── helpers ──────────────────────────────────────────────────────────────────

def _solar_profile(hour: int, day: int) -> float:
    """Return normalised solar irradiance ∈ [0, 1] for a given hour & day."""
    # Seasonal amplitude (peak in summer ≈ day 172)
    seasonal = 0.8 + 0.2 * math.sin(2 * math.pi * (day - 80) / 365)
    # Daily bell curve centred at solar noon (hour 12)
    if 6 <= hour <= 18:
        daily = math.sin(math.pi * (hour - 6) / 12)
    else:
        daily = 0.0
    return seasonal * daily


def _wind_profile(hour: int, day: int, rng: np.random.Generator) -> float:
    """Return normalised wind capacity factor ∈ [0, 1]."""
    # Base wind: slightly higher at night & in winter
    base = 0.3 + 0.15 * math.cos(2 * math.pi * hour / 24)
    seasonal = 1.0 + 0.25 * math.cos(2 * math.pi * (day - 1) / 365)
    noise = rng.normal(0, 0.08)
    return float(np.clip(base * seasonal + noise, 0.0, 1.0))


def _demand_profile(hour: int, day: int, rng: np.random.Generator) -> float:
    """Return normalised demand ∈ [0, 1]."""
    # Double hump: morning 8-10 and evening 18-21
    morning = 0.7 * math.exp(-0.5 * ((hour - 9) / 2) ** 2)
    evening = 1.0 * math.exp(-0.5 * ((hour - 19) / 2) ** 2)
    base = 0.3
    seasonal = 1.0 + 0.15 * math.sin(2 * math.pi * (day - 172) / 365)  # summer peak
    noise = rng.normal(0, 0.03)
    return float(np.clip((base + morning + evening) * seasonal + noise, 0.1, 1.0))


def _grid_price(hour: int) -> tuple[float, float]:
    """Return (buy_price, sell_price) in $/MWh for a given hour."""
    # Time-of-use pricing
    if 7 <= hour <= 11 or 17 <= hour <= 21:
        buy = 120.0  # peak
    elif 11 < hour < 17:
        buy = 80.0   # mid-peak
    else:
        buy = 45.0   # off-peak
    sell = buy * 0.6  # feed-in tariff is lower
    return buy, sell


# ── grid config ──────────────────────────────────────────────────────────────

class GridConfig:
    """Tuneable parameters for the microgrid."""

    SOLAR_PEAK_MW: float = 5.0
    WIND_PEAK_MW: float = 3.0
    BATTERY_CAPACITY_MWH: float = 10.0
    BATTERY_MAX_CHARGE_MW: float = 2.5
    BATTERY_MAX_DISCHARGE_MW: float = 2.5
    BATTERY_EFFICIENCY: float = 0.92       # round-trip √η per half-cycle
    BATTERY_INITIAL_SOC: float = 0.5
    GRID_MAX_IMPORT_MW: float = 8.0
    GRID_MAX_EXPORT_MW: float = 4.0
    PEAK_DEMAND_MW: float = 7.0
    CARBON_INTENSITY_KG_PER_MWH: float = 450.0   # avg grid carbon
    EPISODE_LENGTH_HOURS: int = 168        # 1 week

    # Reward weights (all non-negative, higher = more important)
    W_COST: float = 1.0
    W_CARBON: float = 0.3
    W_UNMET: float = 5.0      # heavy penalty for unmet demand


# ── environment ──────────────────────────────────────────────────────────────

class SmartGridEnvironment(Environment):
    """
    OpenEnv-compatible Smart Grid microgrid RL environment.

    Each episode simulates one week (168 hourly steps) starting from a
    random day of the year.  The agent receives a SmartGridObservation and
    must return a SmartGridAction each hour.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: GridConfig | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.cfg = config or GridConfig()
        self._rng = np.random.default_rng()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Episode accumulators
        self._total_cost = 0.0
        self._total_carbon = 0.0
        self._total_unmet = 0.0

        # Battery
        self._soc = self.cfg.BATTERY_INITIAL_SOC

        # Time
        self._start_day = 1
        self._hour = 0

    # ── reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SmartGridObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._start_day = int(self._rng.integers(1, 366))
        self._hour = 0
        self._soc = self.cfg.BATTERY_INITIAL_SOC
        self._total_cost = 0.0
        self._total_carbon = 0.0
        self._total_unmet = 0.0

        return self._make_observation(first=True)

    # ── step ─────────────────────────────────────────────────────────────

    def step(
        self,
        action: SmartGridAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SmartGridObservation:
        self._state.step_count += 1
        hour, day = self._current_time()

        # --- available generation ---
        solar_avail = _solar_profile(hour, day) * self.cfg.SOLAR_PEAK_MW
        wind_avail = _wind_profile(hour, day, self._rng) * self.cfg.WIND_PEAK_MW
        demand = _demand_profile(hour, day, self._rng) * self.cfg.PEAK_DEMAND_MW
        buy_price, sell_price = _grid_price(hour)

        # --- apply agent actions ---
        solar_dispatch = action.solar_dispatch * solar_avail
        wind_dispatch = action.wind_dispatch * wind_avail
        renewable_total = solar_dispatch + wind_dispatch

        # Battery
        batt_power = self._apply_battery(action.battery_action)

        # Grid exchange (positive = buying)
        grid_power = action.grid_exchange * (
            self.cfg.GRID_MAX_IMPORT_MW if action.grid_exchange >= 0 else self.cfg.GRID_MAX_EXPORT_MW
        )

        # --- balance ---
        supply = renewable_total - batt_power + grid_power  # batt_power>0 means charging (consuming)
        unmet = max(0.0, demand - supply)
        excess = max(0.0, supply - demand)

        # --- costs ---
        if grid_power >= 0:
            energy_cost = grid_power * buy_price / 1000.0   # $/MWh → $/kWh-ish kept in $
        else:
            energy_cost = grid_power * sell_price / 1000.0   # negative = revenue

        carbon = max(0.0, grid_power) * self.cfg.CARBON_INTENSITY_KG_PER_MWH / 1000.0   # per-step

        # --- accumulators ---
        self._total_cost += energy_cost
        self._total_carbon += carbon
        self._total_unmet += unmet

        # --- reward ---
        reward = -(
            self.cfg.W_COST * energy_cost
            + self.cfg.W_CARBON * carbon
            + self.cfg.W_UNMET * unmet
        )

        self._hour += 1
        done = self._hour >= self.cfg.EPISODE_LENGTH_HOURS

        return SmartGridObservation(
            hour_of_day=hour,
            day_of_year=day,
            solar_available=solar_avail,
            wind_available=wind_avail,
            demand=demand,
            battery_soc=self._soc,
            battery_capacity=self.cfg.BATTERY_CAPACITY_MWH,
            grid_buy_price=buy_price,
            grid_sell_price=sell_price,
            solar_dispatched=solar_dispatch,
            wind_dispatched=wind_dispatch,
            battery_power=batt_power,
            grid_power=grid_power,
            energy_cost=energy_cost,
            unmet_demand=unmet,
            excess_energy=excess,
            carbon_emissions=carbon,
            total_cost=self._total_cost,
            total_carbon=self._total_carbon,
            total_unmet=self._total_unmet,
            step_number=self._state.step_count,
            done=done,
            reward=reward,
            metadata={
                "hour": hour,
                "day": day,
                "step": self._state.step_count,
            },
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _current_time(self) -> tuple[int, int]:
        total_hours = self._hour
        day = (self._start_day + total_hours // 24 - 1) % 365 + 1
        hour = total_hours % 24
        return hour, day

    def _apply_battery(self, action_val: float) -> float:
        """
        Apply battery charge/discharge.  Returns actual power in MW.
        Positive = charging (consuming energy), negative = discharging (providing energy).
        """
        cfg = self.cfg
        if action_val >= 0:
            # Charging
            max_charge = min(
                cfg.BATTERY_MAX_CHARGE_MW,
                (1.0 - self._soc) * cfg.BATTERY_CAPACITY_MWH,  # room in battery
            )
            power = action_val * max_charge
            energy_stored = power * cfg.BATTERY_EFFICIENCY
            self._soc += energy_stored / cfg.BATTERY_CAPACITY_MWH
        else:
            # Discharging
            max_discharge = min(
                cfg.BATTERY_MAX_DISCHARGE_MW,
                self._soc * cfg.BATTERY_CAPACITY_MWH,
            )
            power = action_val * max_discharge   # negative
            energy_out = abs(power) * cfg.BATTERY_EFFICIENCY
            self._soc -= energy_out / cfg.BATTERY_CAPACITY_MWH

        self._soc = float(np.clip(self._soc, 0.0, 1.0))
        return power

    def _make_observation(self, first: bool = False) -> SmartGridObservation:
        """Build the initial observation on reset."""
        hour, day = self._current_time()
        solar_avail = _solar_profile(hour, day) * self.cfg.SOLAR_PEAK_MW
        wind_avail = _wind_profile(hour, day, self._rng) * self.cfg.WIND_PEAK_MW
        demand = _demand_profile(hour, day, self._rng) * self.cfg.PEAK_DEMAND_MW
        buy_price, sell_price = _grid_price(hour)

        return SmartGridObservation(
            hour_of_day=hour,
            day_of_year=day,
            solar_available=solar_avail,
            wind_available=wind_avail,
            demand=demand,
            battery_soc=self._soc,
            battery_capacity=self.cfg.BATTERY_CAPACITY_MWH,
            grid_buy_price=buy_price,
            grid_sell_price=sell_price,
            step_number=0,
            done=False,
            reward=0.0,
            metadata={"hour": hour, "day": day, "step": 0},
        )

    @property
    def state(self) -> State:
        return self._state

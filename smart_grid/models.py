# Smart Grid RL Environment – Data Models
# ==========================================
# Action and Observation types for a microgrid energy-management RL task.

from typing import Dict, List, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class SmartGridAction(Action):
    """
    Agent's control decisions for one timestep.

    All values are in [0, 1] and interpreted as *fractions* of available
    capacity so the agent never needs to know absolute MW numbers.
    """

    solar_dispatch: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of available solar generation to dispatch (0-1)",
    )
    wind_dispatch: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of available wind generation to dispatch (0-1)",
    )
    battery_action: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Battery action: negative=discharge, positive=charge, magnitude=fraction of max rate",
    )
    grid_exchange: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Grid exchange: negative=sell to grid, positive=buy from grid, magnitude=fraction of max",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class SmartGridObservation(Observation):
    """
    Full snapshot of the microgrid sent to the agent after each step.
    """

    # --- Time context ---
    hour_of_day: int = Field(default=0, ge=0, le=23, description="Current hour (0-23)")
    day_of_year: int = Field(default=1, ge=1, le=365, description="Day of year (1-365)")

    # --- Renewable availability (MW, before curtailment) ---
    solar_available: float = Field(default=0.0, description="Available solar generation (MW)")
    wind_available: float = Field(default=0.0, description="Available wind generation (MW)")

    # --- Demand ---
    demand: float = Field(default=0.0, description="Current electricity demand (MW)")

    # --- Battery state ---
    battery_soc: float = Field(default=0.5, ge=0.0, le=1.0, description="Battery state-of-charge (0-1)")
    battery_capacity: float = Field(default=0.0, description="Battery total capacity (MWh)")

    # --- Grid price ---
    grid_buy_price: float = Field(default=0.0, description="Price to buy electricity from main grid ($/MWh)")
    grid_sell_price: float = Field(default=0.0, description="Price received when selling to main grid ($/MWh)")

    # --- Previous-step actuals ---
    solar_dispatched: float = Field(default=0.0, description="Solar actually dispatched last step (MW)")
    wind_dispatched: float = Field(default=0.0, description="Wind actually dispatched last step (MW)")
    battery_power: float = Field(default=0.0, description="Battery power last step (MW, +charge/-discharge)")
    grid_power: float = Field(default=0.0, description="Grid power last step (MW, +buy/-sell)")

    # --- Costs / metrics ---
    energy_cost: float = Field(default=0.0, description="Energy cost/revenue for last step ($)")
    unmet_demand: float = Field(default=0.0, description="Unmet demand last step (MW)")
    excess_energy: float = Field(default=0.0, description="Curtailed / wasted energy last step (MW)")
    carbon_emissions: float = Field(default=0.0, description="CO2 emissions from grid power (kg)")

    # --- Cumulative episode stats ---
    total_cost: float = Field(default=0.0, description="Cumulative energy cost this episode ($)")
    total_carbon: float = Field(default=0.0, description="Cumulative CO2 emissions this episode (kg)")
    total_unmet: float = Field(default=0.0, description="Cumulative unmet demand this episode (MWh)")
    step_number: int = Field(default=0, ge=0, description="Current step within episode")

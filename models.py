"""
Smart Grid Demand Response — Data Models (Phase 2)
====================================================
Action, Observation types for a city power grid demand response environment.
Phase 2 adds: battery storage fields, tripped load tracking.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation — what the agent SEES each step
# ---------------------------------------------------------------------------
class Observation(BaseModel):
    """
    Rich observation with both structured data AND a natural-language
    situation report so LLM agents can reason about grid state.
    """

    # ── Time context ──
    hour: int = Field(default=0, ge=0, le=23, description="Current hour (0-23)")
    day: int = Field(default=1, ge=1, le=365, description="Day of year (1-365)")
    step_number: int = Field(default=0, ge=0, description="Current step in episode")

    # ── Grid health ──
    grid_frequency_hz: float = Field(
        default=50.0,
        description="Grid frequency in Hz. Normal=50.0, Warning<49.5, Critical<49.2, Blackout<48.5"
    )
    alert_level: str = Field(
        default="normal",
        description="Grid alert: 'normal', 'warning', 'critical', 'emergency', 'blackout'"
    )

    # ── Supply & Demand ──
    total_demand_mw: float = Field(default=0.0, description="Total electricity demand (MW)")
    total_supply_mw: float = Field(default=0.0, description="Total available supply (MW)")
    solar_output_mw: float = Field(default=0.0, description="Current solar generation (MW)")
    wind_output_mw: float = Field(default=0.0, description="Current wind generation (MW)")
    thermal_output_mw: float = Field(default=0.0, description="Thermal/coal generation (MW)")
    supply_deficit_mw: float = Field(default=0.0, description="Supply shortfall (MW), 0 if surplus")

    # ── Battery Storage ──
    battery_soc_pct: float = Field(
        default=50.0,
        ge=0.0, le=100.0,
        description="Battery state of charge (0-100%)"
    )
    battery_capacity_mwh: float = Field(default=50.0, description="Total battery capacity (MWh)")
    battery_hours_remaining: float = Field(
        default=1.0,
        description="Hours of discharge available at max rate (25MW)"
    )

    # ── Cascading Failures ──
    tripped_loads: List[str] = Field(
        default_factory=list,
        description="Load IDs that have been auto-disconnected due to low frequency"
    )
    low_freq_consecutive_steps: int = Field(
        default=0,
        description="Consecutive steps below 49.2Hz (cascade triggers at 2)"
    )

    # ── Weather ──
    weather: str = Field(default="clear", description="Weather: clear, cloudy, heatwave, storm, monsoon")
    temperature_c: float = Field(default=30.0, description="Temperature in Celsius")

    # ── Controllable loads ──
    loads: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of controllable loads. Each: "
            '{"id": str, "name": str, "current_mw": float, "reducible_mw": float, '
            '"priority": str, "discomfort_factor": float, "curtailed_this_episode": int, '
            '"tripped": bool, "reconnect_in": int}'
        )
    )

    # ── Previous step results ──
    total_curtailed_mw: float = Field(default=0.0, description="MW curtailed last step")
    curtailment_cost_inr: float = Field(default=0.0, description="Cost of curtailment last step (₹)")
    renewable_utilization_pct: float = Field(default=0.0, description="% of supply from renewables")
    tou_rate_inr_per_kwh: float = Field(default=8.0, description="Current time-of-use tariff (₹/kWh)")

    # ── Cumulative episode stats ──
    cumulative_cost_inr: float = Field(default=0.0, description="Total curtailment cost this episode (₹)")
    cumulative_discomfort: float = Field(default=0.0, description="Total discomfort this episode")
    blackout_occurred: bool = Field(default=False, description="Has a blackout happened this episode?")
    cascade_events: int = Field(default=0, description="Number of auto-disconnect events this episode")

    # ── Natural language situation report (LLM-native) ──
    situation_report: str = Field(
        default="",
        description="Human-readable strategic grid briefing for LLM reasoning"
    )

    # ── RL / OpenEnv standard fields ──
    reward: float = Field(default=0.0, description="Reward for this step (higher is better)")
    done: bool = Field(default=False, description="Is the episode over?")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rich metadata: frequency_history, tripped_loads, battery_soc, fairness_gini, projected_deficit, etc."
    )


# ---------------------------------------------------------------------------
# Action — what the agent DOES each step
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """
    Agent's demand response decision.
    
    - Curtail specific loads (reduce their MW draw)
    - Optionally charge or discharge the 50MWh battery
    
    Example:
        {
          "curtailments": {"steel_plant": 20.0, "cement_factory": 15.0},
          "battery_action": "discharge",
          "battery_mw": 20.0
        }
    """
    curtailments: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Map of load_id → MW to reduce. "
            "Example: {'steel_plant': 20.0, 'textile_mill': 10.0}"
        )
    )
    battery_action: str = Field(
        default="idle",
        description="Battery action: 'charge' (store surplus), 'discharge' (inject supply), or 'idle'"
    )
    battery_mw: float = Field(
        default=0.0,
        ge=0.0,
        le=25.0,
        description="MW to charge or discharge (max 25MW, min 0)"
    )

"""Smart Grid Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SmartGridAction, SmartGridObservation


class SmartGridEnv(
    EnvClient[SmartGridAction, SmartGridObservation, State]
):
    """
    Client for the Smart Grid Microgrid RL Environment.

    Connects to a running SmartGridEnvironment server over HTTP/WebSocket
    and exposes the standard ``reset()`` / ``step()`` API.

    Example:
        >>> with SmartGridEnv(base_url="http://localhost:8000") as env:
        ...     obs = env.reset()
        ...     action = SmartGridAction(
        ...         solar_dispatch=1.0,
        ...         wind_dispatch=1.0,
        ...         battery_action=-0.3,
        ...         grid_exchange=0.1,
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.demand, result.reward)
    """

    def _step_payload(self, action: SmartGridAction) -> Dict[str, Any]:
        return {
            "solar_dispatch": action.solar_dispatch,
            "wind_dispatch": action.wind_dispatch,
            "battery_action": action.battery_action,
            "grid_exchange": action.grid_exchange,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SmartGridObservation]:
        obs_data = payload.get("observation", {})
        observation = SmartGridObservation(
            hour_of_day=obs_data.get("hour_of_day", 0),
            day_of_year=obs_data.get("day_of_year", 1),
            solar_available=obs_data.get("solar_available", 0.0),
            wind_available=obs_data.get("wind_available", 0.0),
            demand=obs_data.get("demand", 0.0),
            battery_soc=obs_data.get("battery_soc", 0.5),
            battery_capacity=obs_data.get("battery_capacity", 0.0),
            grid_buy_price=obs_data.get("grid_buy_price", 0.0),
            grid_sell_price=obs_data.get("grid_sell_price", 0.0),
            solar_dispatched=obs_data.get("solar_dispatched", 0.0),
            wind_dispatched=obs_data.get("wind_dispatched", 0.0),
            battery_power=obs_data.get("battery_power", 0.0),
            grid_power=obs_data.get("grid_power", 0.0),
            energy_cost=obs_data.get("energy_cost", 0.0),
            unmet_demand=obs_data.get("unmet_demand", 0.0),
            excess_energy=obs_data.get("excess_energy", 0.0),
            carbon_emissions=obs_data.get("carbon_emissions", 0.0),
            total_cost=obs_data.get("total_cost", 0.0),
            total_carbon=obs_data.get("total_carbon", 0.0),
            total_unmet=obs_data.get("total_unmet", 0.0),
            step_number=obs_data.get("step_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

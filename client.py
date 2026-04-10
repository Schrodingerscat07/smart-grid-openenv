# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Grid Demand Response Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import Action, Observation


class SmartGridEnv(
    EnvClient[Action, Observation, State]
):
    """
    Client for the Smart Grid Demand Response Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SmartGridEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.situation_report)
        ...
        ...     result = client.step(Action(curtailments={"steel_plant": 10.0}))
        ...     print(result.observation.grid_frequency_hz)
    """

    def _step_payload(self, action: Action) -> Dict:
        """
        Convert Action to JSON payload for step message.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        """
        Parse server response into StepResult[Observation].
        """
        obs_data = payload.get("observation", {})
        
        # Merge reward/done from top level if they exist there
        if "reward" in payload:
            obs_data["reward"] = payload["reward"]
        if "done" in payload:
            obs_data["done"] = payload["done"]
            
        observation = Observation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

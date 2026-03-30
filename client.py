"""PayOps Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from payops_env.models import PayOpsAction, PayOpsObservation


class PayOpsEnv(EnvClient[PayOpsAction, PayOpsObservation]):
    """
    Client for the PayOps Payment Operations Incident Response Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions.

    Example::

        with PayOpsEnv(base_url="http://localhost:7860") as client:
            result = client.reset()
            while not result.done:
                action = PayOpsAction(
                    action_type="approve",
                    transaction_id=result.observation.transaction_id,
                )
                result = client.step(action)
            print("Score:", result.observation.cumulative_reward)
    """

    def _step_payload(self, action: PayOpsAction) -> Dict[str, Any]:
        """Convert PayOpsAction to the official openenv wire format."""
        return {
            "action": action.model_dump(exclude_none=True),
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PayOpsObservation]:
        """Parse server response into StepResult[PayOpsObservation]."""
        obs_data = payload.get("observation", payload)  # support wrapped or flat
        observation = PayOpsObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server state response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

# OpenEnv Course Reference (raun/openenv-course)

This file is a distilled reference for agents building OpenEnv environments.
Source: https://github.com/raun/openenv-course

## Module Structure
- module-1: Why OpenEnv?
- module-2: Using Existing Environments
- module-3: Deploying Environments (`openenv push`)
- module-4: Building Your Own Environment
- module-5: Training with OpenEnv + TRL (GRPO)

## Canonical Directory Structure (Module 4)
```
my_env/
├── models.py              ← Types: Action, Observation, State (Pydantic)
├── client.py              ← HTTP/WebSocket client
├── inference.py           ← Baseline agent script
├── server/
│   ├── environment.py     ← Game logic (reset, step, state)
│   ├── app.py             ← FastAPI server
│   └── __init__.py
├── openenv.yaml           ← Manifest
├── Dockerfile             ← Container definition
└── pyproject.toml         ← Package metadata
```

## 3-Method Interface
Every standard OpenEnv environment uses:
- `reset()` → returns initial Observation
- `step(action)` → returns Observation (with reward, done)
- `state` property → returns State

## Key Types (from openenv.core)
```python
from openenv.core.env_server import Action, Observation, State, Environment
from openenv.core.env_server.http_server import create_app
from openenv.core.env_client import EnvClient
```

- Action, Observation, State are Pydantic BaseModel subclasses
- Observation already has `done: bool` and `reward: Optional[float]`
- State already has `episode_id: Optional[str]` and `step_count: int`

## FastAPI Server Setup (app.py)
```python
from openenv.core.env_server.http_server import create_app
app = create_app(MyEnvironment, MyAction, MyObservation, env_name="my_env")
```

## GRPO Training (Module 5)
- Uses TRL's GRPOTrainer with rollout functions
- Rollout function: plays episodes, returns prompt/completion/reward data
- Reward functions can be multiple (shaping signals + final outcome)
- Key: No value model needed (group provides baseline)

## Deployment
```bash
openenv validate <env_name>   # Check spec compliance
openenv push --repo-id user/env-name  # Push to HF Spaces
```

## Grading / Tasks
- Environments should have 3+ tasks with difficulty range (easy/medium/hard)
- Each task has a `grade()` method returning 0.0-1.0
- Graders must be deterministic and reproducible

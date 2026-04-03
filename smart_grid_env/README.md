# Smart Grid Demand Response Environment

This environment simulates a city power grid where an AI agent must manage demand response to prevent blackouts during peak hours.

## Features
- **Realistic Demand Curves**: Evening peaks and morning ramps.
- **Renewable Integration**: Solar and wind variable output.
- **Grid Stability Model**: Frequency tracking based on supply/demand imbalance.
- **Task Graders**: 3 levels of difficulty (Easy, Medium, Hard).

## How to use
Run the FastAPI server locally:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

Run the baseline evaluation:
```bash
python inference.py
```

## Structure
- `env/`: Core logic (simulation, models, tasks).
- `app.py`: API layer.
- `inference.py`: Evaluation script.
- `openenv.yaml`: OpenEnv metadata.

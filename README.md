---
title: Smart Grid Demand Response
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
base_path: /web
---

# ⚡ Smart Grid Demand Response (OpenEnv)

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Maybe-Heisenberg-07/smart-grid-demand-response)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/huggingface/openenv)

> **The first demand response reinforcement learning environment natively designed for LLM Agents.**

---

## 🌍 The Problem Statement

Every city grid faces a major crisis daily: **unpredictable demand vs. intermittent renewables**.

Imagine it's 6:00 PM on a 45°C day in Delhi. 20 million ACs switch on. Solar drops to zero. Grid frequency plummets. **In 15 minutes, there will be a blackout unless someone acts.**

Massive power operators like **Adani Power**, **Tata Power**, or the **Power System Operation Corporation (POSOCO) in India** face this exact challenge. They have to balance the electricity flow through the grid to keep it exactly at **50Hz**. If people use more power than generators produce, the frequency drops. If the frequency drops too low, transformers explode and the entire city goes dark.

Currently, automated systems use complex numeric models — they balance numbers, but fail to understand real-world context: an algorithm doesn't inherently understand that turning off a **Hospital** is a catastrophe, while briefly reducing a **Steel Plant** is acceptable.

---

## 💎 Our Unique Novelty: An Agentic Approach

Demand response isn't a new problem. Existing RL environments for power grids (like CityLearn or Grid2Op) use **flat numeric vectors** — arrays like `[50.2, 280.3, 45.1]`.

**This makes them impossible for Large Language Models (LLMs) to reason about.**

**Our environment is the very first demand response simulator natively designed for LLM agents.** Here's what makes it a breakthrough:

### 1. Situation Reports, Not Vectors
Instead of obscure numbers, the agent receives a naturally written **Strategic Briefing**:
> *"⚠️ WARNING: Grid frequency at 49.6Hz and falling. Evening peak in 45 minutes. Solar output declining. Steel plant running at full capacity (80MW reducible by 32MW). Hospital on backup generator — DO NOT curtail."*

### 2. Cascading Failure Mechanics
If grid frequency falls below **49.0Hz**, loads automatically start disconnecting in a brutal cascade. The agent must think ahead: *"If I don't curtail the factories now, the hospital goes dark in 3 steps."*

### 3. Constrained Ethical Decision-Making
The grader strictly evaluates the agent on **fairness** and **Critical Infrastructure Protection**. An agent that keeps the grid alive but repeatedly shuts down hospitals will spectacularly fail.

---

## 🎮 How the Solution Works

The AI operator has two main tools to combat a grid crisis:

1. **Load Curtailment:** Selectively instructing specific facilities (like factories) to lower their power usage.
2. **Battery Energy Storage (BESS):** Managing a massive 50MWh grid-scale battery — charge during cheap off-peak hours, discharge during emergencies.

The AI must coordinate these actions over an extended timeframe while navigating weather, consumer demands, and economic cost.

---

## 🏆 The 5 Mission Scenarios

| Task | Difficulty | Steps | Focus |
| :--- | :--- | :---: | :--- |
| **Peak Survival** | Easy | 12 | Survive a 3-hour evening spike |
| **Daily Balance** | Medium | 24 | 24h stability & cost optimization |
| **Extreme Event** | Hard | 48 | 48h heatwave crisis (Delhi style) |
| **Monsoon Crisis** | Medium-Hard | 24 | Zero solar, erratic wind, heavy BESS |
| **Renewable Transition** | Expert | 72 | Coal retired — 100% green + battery |

---

## 🚀 Quick Start

### Connect to the Live Space
```python
from client import SmartGridEnv
from models import Action

# Connect to the running HF Space
async with SmartGridEnv(base_url="https://Maybe-Heisenberg-07-smart-grid-demand-response.hf.space") as env:
    result = await env.reset()
    print(result.observation.situation_report)

    result = await env.step(Action(
        curtailments={"steel_plant": 15.0},
        battery_action="discharge",
        battery_mw=20.0
    ))
```

### Run Locally
```bash
# Clone the repo
git clone https://github.com/Schrodingerscat07/smart-grid-openenv.git
cd smart-grid-openenv

# Install dependencies
pip install -e .

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or use Docker
docker build -t smart-grid .
docker run -d -p 7860:7860 smart-grid
```

### Install as a Package
```bash
pip install git+https://huggingface.co/spaces/Maybe-Heisenberg-07/smart-grid-demand-response
```

---

## 🏗️ Architecture

```
smart-grid-openenv/            # Repo Root = OpenEnv Environment
├── server/
│   ├── app.py                 # FastAPI entry point (create_app)
│   ├── grid_env.py            # Main Environment class (reset/step/state)
│   ├── simulator.py           # Physics engine (frequency, demand, weather)
│   ├── tasks.py               # 5 task definitions + deterministic graders
│   └── weather.py             # Weather system (heatwave, monsoon, etc.)
├── models.py                  # Pydantic Action & Observation types
├── client.py                  # WebSocket client (EnvClient subclass)
├── inference.py               # LLM baseline agent (Phase 2 compliant)
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile                 # Multi-stage Docker build
└── pyproject.toml             # Dependencies & entry points
```

---

## 🕵️ Judging & Testing Guide

If you are evaluating this project, try to "break" the grid to see our physics engine at work:

### The Blackout Challenge
1. Open the **Web UI** → select **"extreme_event"**.
2. Click **Reset**.
3. **Do absolutely nothing.** Leave curtailment at `{}` and battery on `idle`.
4. Click **Step** continuously.
5. Watch the frequency plummet, cascading failures trigger, and the **Situation Report** describe neighborhoods going dark!

---

## 📊 Proof of Variance

A good environment must have score variance. We mathematically proved the solvability:

| Strategy | Score | Outcome |
| :--- | :---: | :--- |
| **Do Nothing** | `0.001` | Instant blackout — total failure |
| **Random Actions** | `0.05–0.19` | Unstable, frequent cascades |
| **Basic Heuristic** | `0.21` | Survives but poor efficiency |
| **Smart Oracle** | `0.65+` | Professional grid management |

Incompetent agents crash and burn; intelligent agents thrive. ⚡

---

## 📋 Environment Variables

| Variable | Required | Description |
| :--- | :---: | :--- |
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `HF_TOKEN` | Yes | Hugging Face / API key |

**Built for the [Meta PyTorch Hackathon × Scaler](https://pytorch.org/) — OpenEnv Track.** ⚡

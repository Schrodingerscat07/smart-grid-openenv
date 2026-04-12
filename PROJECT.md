# Smart Grid Demand Response — Project Bible

> **One-liner**: First demand response RL environment designed for LLM agents with natural language situational awareness.

---

## 📋 Problem Statement

Every city grid faces the same crisis daily: unpredictable demand vs intermittent renewables. At 6pm on a 45°C day in Delhi, 20 million ACs switch on. Solar drops to zero. Grid frequency plummets. Blackout in 15 minutes unless someone acts.

**Existing RL environments** for demand response (CityLearn, Grid2Op) use flat numeric vectors — they're designed for PPO/SAC agents that see `[50.2, 280.3, 45.1, ...]`. An LLM can't reason about those numbers.

**Our environment is the first demand response simulator designed specifically for LLM agents.** The key innovations:

1. **Situation Reports, Not Vectors** — Each observation includes a human-readable grid status briefing: *"⚠️ WARNING: Grid frequency at 49.6Hz and falling. Evening peak in 45 minutes. Solar output declining. Steel plant running at full capacity (80MW reducible by 32MW). Hospital on backup generator — DO NOT curtail."*

2. **Constrained Decision-Making with Ethical Stakes** — The agent must reason about trade-offs that require *understanding*, not just optimization: curtailing a hospital vs a factory, fairness across neighborhoods, balancing economic cost vs human comfort.

3. **Cascading Failure Mechanics** — If frequency drops below 49.0Hz, loads auto-disconnect in a cascade. The agent must think ahead: "If I don't curtail now, I'll lose the hospital in 3 steps." This creates genuine strategic depth that challenges frontier LLMs.

4. **Adversarial Weather Events** — Cyclones, heatwaves, and monsoons create crisis scenarios where the agent must adapt its strategy in real-time, not just follow learned patterns.

5. **Multi-Objective Grading with Fairness** — Unlike standard RL environments with a single reward, we grade on stability × cost × fairness × comfort simultaneously. An agent that keeps the grid alive but bankrupts consumers scores poorly.

---

## 🏆 Evaluation Criteria

| Parameter | Weight | Description |
|---|---|---|
| **Real-world utility** | 30% | Does the environment model a genuine task? Would someone actually use this to train or evaluate agents? |
| **Task & grader quality** | 25% | Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression? |
| **Environment design** | 20% | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries. |
| **Code quality & spec compliance** | 15% | Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works. |
| **Creativity & novelty** | 10% | Novel problem domain, interesting mechanics, clever reward design, original approach. |

### Scoring Breakdown

#### Real-world utility (30%)
- 0–5: Toy/artificial problem with no practical application
- 6–15: Valid domain but shallow modeling of the real task
- 16–25: Good domain modeling, would be useful for agent evaluation
- **26–30: Excellent — fills a real gap, immediate value for the RL/agent community** ← OUR TARGET

#### Task & grader quality (25%)
- 3+ tasks with difficulty range? ✅
- Graders produce scores between 0.0–1.0? ✅
- Graders deterministic and reproducible? ✅
- Hard task genuinely challenges frontier models? ✅

#### Environment design (20%)
- reset() produces clean state? ✅
- Action/observation types well-designed and documented? ✅
- Reward function provides useful varying signal (not just sparse)? ✅
- Episode boundaries sensible? ✅

#### Code quality & spec compliance (15%)
- openenv validate passes? ✅
- docker build && docker run works? ✅
- HF Space deploys and responds? ✅
- Baseline script runs and reproduces scores? ✅

#### Creativity & novelty (10%)
- Domain we haven't seen in OpenEnv before? ✅
- Reward design has interesting properties? ✅
- Clever mechanics that make the environment engaging? ✅

---

## 🔍 Judging Pipeline

### Phase 1: Automated Validation (Pass/Fail Gate)
- HF Space deploys and responds
- OpenEnv spec compliance (`openenv validate`)
- Dockerfile builds and runs
- Baseline inference.py reproduces scores
- 3+ tasks with graders

### Phase 2: Agentic Evaluation (Scored)
- Baseline agent re-run
- Standard Open LLM agent (e.g. Nemotron 3 Super) run against all environments
- Score variance check (graders must give *varying* scores)

### Phase 3: Human Review (Top submissions)
- Meta and Hugging Face engineers review for real-world utility, creativity, exploit checks

### Disqualification Criteria
- ❌ Environment does not deploy or respond
- ❌ Plagiarized or trivially modified existing environments
- ❌ Graders that always return the same score
- ❌ No baseline inference script

---

## 🏗️ Build Plan — Phased Approach

### Phase 1: Core Environment (MVP)
> COMPLETED ✅

### Phase 2: Depth & Realism
> COMPLETED ✅

### Phase 3: Polish & Deploy
> COMPLETED ✅

### Phase 4: Winning Edge (Final Submission)
> STATUS: FINAL SUBMISSION READY 🚀


### Phase 4: Winning Edge (Scope for Improvement)
> Differentiators that push from top-10 to top-3

- [ ] Battery Energy Storage System (agent chooses curtail OR discharge battery)
- [ ] EV charging fleet integration
- [ ] Renewable forecast errors (predicted vs actual)
- [ ] Day-of-week and festival patterns (Diwali demand spike)
- [ ] Historical demand data from real Indian grid (POSOCO/CERC)
- [ ] Advanced fairness metrics (Gini coefficient across neighborhoods)
- [ ] Multi-agent scenario (multiple grid operators coordinating)
- [ ] Visualization dashboard data in metadata

---

## 📁 Canonical Directory Structure

```
smart_grid_env/                      # Root of submission
├── models.py                        # Pydantic Action/Observation (MUST be in root)
├── inference.py                     # LLM baseline agent
├── openenv.yaml                     # Manifest with tasks
├── Dockerfile                       # Multi-stage build
├── pyproject.toml                   # Dependencies
├── README.md                        # Documentation
├── PROJECT.md                       # This file
└── server/
    ├── __init__.py
    ├── app.py                       # FastAPI entry (create_app)
    ├── grid_env.py                  # Main Environment class
    ├── simulator.py                 # Physics engine
    └── tasks.py                     # Task definitions + graders
```

### Rules (from AGENTS.md)
- `models.py` MUST be in root (not in server/)
- `app.py` must have `if __name__ == "__main__": main()` at the end (strict regex check)
- Always run `openenv validate` before pushing
- Set `$env:PYTHONUTF8=1` on Windows before running CLI

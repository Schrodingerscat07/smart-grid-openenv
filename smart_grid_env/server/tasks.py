"""
Task Definitions & Graders (Phase 2)
======================================
Each task defines episode parameters and a grade() method
that returns a deterministic score between 0.0 and 1.0.

Tasks:
1. peak_survival       (EASY)        — 12 steps, survive evening peak
2. daily_balance       (MEDIUM)      — 24 steps, 24-hour balance
3. extreme_event       (HARD)        — 48 steps, heatwave crisis
4. monsoon_crisis      (MEDIUM-HARD) — 24 steps, monsoon storm + battery management
5. renewable_transition (EXPERT)     — 72 steps, 30% less thermal, multi-day planning
"""

from __future__ import annotations
from typing import List, Dict, Any


# ── Base Task ───────────────────────────────────────────────────────────────

class Task:
    name: str = ""
    description: str = ""
    difficulty: str = ""
    episode_steps: int = 12
    start_hour: int = 17
    start_day: int = 150
    forced_weather: str | None = None
    thermal_multiplier: float = 1.0  # For renewable_transition task

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        raise NotImplementedError


# ── Helpers ──────────────────────────────────────────────────────────────────

def _freq_score(freq: float) -> float:
    """Map frequency to [0,1] score."""
    if freq >= 49.8:
        return 1.0
    elif freq >= 49.5:
        return 0.7
    elif freq >= 49.2:
        return 0.3
    elif freq >= 49.0:
        return 0.1
    else:
        return 0.0

def _stability_score(history: List[Dict]) -> float:
    scores = [_freq_score(s.get("grid_frequency_hz", 50.0)) for s in history]
    return sum(scores) / len(scores) if scores else 0.0

def _discomfort_score(history: List[Dict]) -> float:
    total = sum(s.get("avg_discomfort", 0.0) for s in history)
    n = len(history)
    return max(0.0, 1.0 - total / max(1, n))

def _cost_score(history: List[Dict], cap_inr: float) -> float:
    total = sum(s.get("curtailment_cost_inr", 0.0) for s in history)
    return max(0.0, 1.0 - total / cap_inr)

def _blackout_free(history: List[Dict]) -> bool:
    return all(s.get("grid_frequency_hz", 50.0) >= 48.5 for s in history)

def _fairness_score(history: List[Dict], n_steps: int) -> float:
    counts: Dict[str, int] = {}
    for step in history:
        for load_id, details in step.get("per_load_curtailments", {}).items():
            if details.get("actual_mw", 0) > 0:
                counts[load_id] = counts.get(load_id, 0) + 1
    max_allowed = 0.40 * n_steps
    violations = sum(1 for c in counts.values() if c > max_allowed)
    return max(0.0, 1.0 - violations / 10)  # 10 total loads

def _critical_protection_score(history: List[Dict]) -> float:
    critical_ids = {"hospital", "metro_rail"}
    curtailments = 0
    for step in history:
        for cid in critical_ids:
            if step.get("per_load_curtailments", {}).get(cid, {}).get("actual_mw", 0) > 0:
                curtailments += 1
    return max(0.0, 1.0 - curtailments * 0.10)

def _no_cascade_score(history: List[Dict]) -> float:
    total_trips = sum(len(s.get("newly_tripped_loads", [])) for s in history)
    return max(0.0, 1.0 - total_trips * 0.10)

def _cascade_penalty(base_score: float, history: List[Dict]) -> float:
    """Brutal penalty: Halves the score for EVERY load that auto-disconnects."""
    total_trips = sum(len(s.get("newly_tripped_loads", [])) for s in history)
    return base_score * (0.5 ** total_trips)

def _battery_utilization_score(history: List[Dict]) -> float:
    """Score for wise battery use: rewards both charging and discharging."""
    charge_steps = sum(
        1 for s in history
        if s.get("battery_result", {}).get("battery_action") == "charge"
    )
    discharge_steps = sum(
        1 for s in history
        if s.get("battery_result", {}).get("battery_action") == "discharge"
    )
    n = len(history)
    # Reward having used the battery (charge + discharge), penalise pure idle
    used = (charge_steps + discharge_steps) / max(1, n)
    return min(1.0, used * 2.0)  # Scale so ~50% use = 1.0

def _renewable_utilization_score(history: List[Dict], target_pct: float = 50.0) -> float:
    mean_pct = sum(s.get("renewable_pct", 0.0) for s in history) / max(1, len(history))
    return min(1.0, mean_pct / target_pct)


# ── Easy: Peak Survival ─────────────────────────────────────────────────────

class PeakSurvivalTask(Task):
    """
    EASY: Survive one evening peak (5pm–8pm, 12 steps at 15-min intervals).
    Focus: stability and not blacking out. Battery available but not required.
    Grading: 60% stability, 40% blackout-free
    """
    name = "peak_survival"
    description = "Survive a 3-hour evening peak without blackout. Keep grid frequency above 49.5 Hz."
    difficulty = "easy"
    episode_steps = 12
    start_hour = 17
    start_day = 160  # June

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0
        stability = _stability_score(episode_history)
        blackout = 1.0 if _blackout_free(episode_history) else 0.0
        base_score = round(0.60 * stability + 0.40 * blackout, 4)
        return _cascade_penalty(base_score, episode_history)


# ── Medium: Daily Balance ───────────────────────────────────────────────────

class DailyBalanceTask(Task):
    """
    MEDIUM: Balance the grid for 24 hours, minimizing discomfort and cost.
    Grading: 40% stability, 30% discomfort, 20% cost, 10% renewable utilization
    """
    name = "daily_balance"
    description = "Balance grid for 24 hours. Minimize discomfort while maintaining stability."
    difficulty = "medium"
    episode_steps = 24
    start_hour = 0
    start_day = 150  # Summer

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0
        stability = _stability_score(episode_history)
        discomfort = _discomfort_score(episode_history)
        cost = _cost_score(episode_history, cap_inr=500_000)
        renewable = _renewable_utilization_score(episode_history, target_pct=30.0)
        base_score = round(0.40 * stability + 0.30 * discomfort + 0.20 * cost + 0.10 * renewable, 4)
        return _cascade_penalty(base_score, episode_history)


# ── Hard: Extreme Weather Event ─────────────────────────────────────────────

class ExtremeWeatherTask(Task):
    """
    HARD: 48-hour heatwave crisis. Stability, cost, fairness, critical protection.
    Grading: 30% stability, 25% fairness, 20% critical infra, 15% cost, 10% discomfort
    """
    name = "extreme_event"
    description = "Handle a 48-hour heatwave. Balance stability, cost, fairness, and protect critical infrastructure."
    difficulty = "hard"
    episode_steps = 48
    start_hour = 6
    start_day = 155  # Peak summer
    forced_weather = "heatwave"

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0
        n = len(episode_history)
        stability = _stability_score(episode_history)
        fairness = _fairness_score(episode_history, n)
        critical = _critical_protection_score(episode_history)
        cost = _cost_score(episode_history, cap_inr=2_000_000)
        discomfort = _discomfort_score(episode_history)
        base_score = round(
            0.30 * stability + 0.25 * fairness + 0.20 * critical +
            0.15 * cost + 0.10 * discomfort, 4
        )
        return _cascade_penalty(base_score, episode_history)


# ── Medium-Hard: Monsoon Crisis ─────────────────────────────────────────────

class MonsoonCrisisTask(Task):
    """
    MEDIUM-HARD: 24-hour monsoon storm. Solar near zero, wind erratic.
    
    The agent must intelligently charge the battery during wind spikes and
    discharge when wind dies. BESS management is the key skill here.
    
    Grading:
    - 35% frequency stability
    - 25% battery utilization efficiency (charge & discharge wisely)
    - 20% cost minimization
    - 20% no-blackout bonus
    """
    name = "monsoon_crisis"
    description = (
        "Survive a 24-hour monsoon storm. Solar is near zero, wind is erratic. "
        "Manage the battery wisely — charge during wind spikes, discharge when wind dies."
    )
    difficulty = "medium-hard"
    episode_steps = 24
    start_hour = 0
    start_day = 210  # Late July — deep monsoon
    forced_weather = "monsoon"

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0
        stability = _stability_score(episode_history)
        battery_util = _battery_utilization_score(episode_history)
        cost = _cost_score(episode_history, cap_inr=600_000)
        blackout = 1.0 if _blackout_free(episode_history) else 0.0
        base_score = round(
            0.35 * stability + 0.25 * battery_util + 0.20 * cost + 0.20 * blackout, 4
        )
        return _cascade_penalty(base_score, episode_history)


# ── Expert: Renewable Transition ─────────────────────────────────────────────

class RenewableTransitionTask(Task):
    """
    EXPERT: 72-hour multi-day episode (3 days).
    
    Scenario: A coal plant has been retired — thermal capacity is reduced 30%.
    The agent must manage solar + wind + battery across multiple day-night cycles.
    Fairness constraints are strict; critical infra must never lose power.
    
    This task demands long-horizon thinking: battery must be charged during the
    day when solar is cheap and discharged at night when thermal is scarce.
    
    Grading:
    - 25% frequency stability
    - 25% renewable utilization (must use >50% renewables on average)
    - 20% fairness (no load curtailed more than 40% of steps)
    - 15% cost efficiency
    - 15% no cascading failures
    """
    name = "renewable_transition"
    description = (
        "3-day simulation with 30% less thermal capacity (coal plant retired). "
        "Rely on renewables + battery + smart curtailment across day-night cycles. "
        "Fairness and cascading failure avoidance are strictly graded."
    )
    difficulty = "expert"
    episode_steps = 72  # 3 × 24 hours
    start_hour = 0
    start_day = 90   # April — spring, good solar, some wind
    thermal_multiplier = 0.70  # 30% less thermal capacity

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0
        n = len(episode_history)
        stability = _stability_score(episode_history)
        renewable = _renewable_utilization_score(episode_history, target_pct=50.0)
        fairness = _fairness_score(episode_history, n)
        cost = _cost_score(episode_history, cap_inr=5_000_000)
        no_cascade = _no_cascade_score(episode_history)
        base_score = round(
            0.25 * stability + 0.25 * renewable + 0.20 * fairness +
            0.15 * cost + 0.15 * no_cascade, 4
        )
        return _cascade_penalty(base_score, episode_history)


# ── Task registry ────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "peak_survival":       PeakSurvivalTask(),
    "daily_balance":       DailyBalanceTask(),
    "extreme_event":       ExtremeWeatherTask(),
    "monsoon_crisis":      MonsoonCrisisTask(),
    "renewable_transition": RenewableTransitionTask(),
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: '{name}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]

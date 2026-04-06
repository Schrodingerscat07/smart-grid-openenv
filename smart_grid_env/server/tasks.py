"""
Task Definitions & Graders
============================
Each task defines episode parameters and a grade() method
that returns a deterministic score between 0.0 and 1.0.

Tasks:
1. peak_survival (EASY)    — Survive a 3-hour evening peak without blackout
2. daily_balance (MEDIUM)  — Balance grid for 24 hours, minimize discomfort
3. extreme_event (HARD)    — Handle a heatwave crisis over 48 hours with fairness
"""

from __future__ import annotations
from typing import List, Dict, Any


# ── Base Task ───────────────────────────────────────────────────────────────

class Task:
    """Base class for all tasks."""
    name: str = ""
    description: str = ""
    difficulty: str = ""
    episode_steps: int = 12
    start_hour: int = 17
    start_day: int = 150  # Late May (summer in India)
    forced_weather: str | None = None  # Override weather for specific tasks

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        """Returns deterministic score 0.0 (failed) to 1.0 (perfect)."""
        raise NotImplementedError


# ── Easy: Peak Survival ─────────────────────────────────────────────────────

class PeakSurvivalTask(Task):
    """
    EASY: Survive one evening peak (5pm–8pm, 3 hours = 12 steps at 15-min intervals).

    The agent just needs to avoid blackout. Partial credit for
    maintaining good frequency. No fairness or cost optimization needed.

    Grading:
    - 60% weight on frequency stability (staying above 49.5 Hz)
    - 40% weight on avoiding blackout (staying above 49.0 Hz)
    """
    name = "peak_survival"
    description = "Survive a 3-hour evening peak without blackout. Keep grid frequency above 49.5 Hz."
    difficulty = "easy"
    episode_steps = 12         # 3 hours at 15-min intervals
    start_hour = 17            # 5pm — peak begins
    start_day = 160            # June — hot summer

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0

        # Frequency stability score
        freq_scores = []
        blackout_free = True
        for step in episode_history:
            freq = step.get("grid_frequency_hz", 50.0)
            if freq >= 49.8:
                freq_scores.append(1.0)
            elif freq >= 49.5:
                freq_scores.append(0.7)
            elif freq >= 49.2:
                freq_scores.append(0.3)
            elif freq >= 49.0:
                freq_scores.append(0.1)
            else:
                freq_scores.append(0.0)
                blackout_free = False

        stability = sum(freq_scores) / len(freq_scores)
        blackout_score = 1.0 if blackout_free else 0.0

        return round(0.6 * stability + 0.4 * blackout_score, 4)


# ── Medium: Daily Balance ───────────────────────────────────────────────────

class DailyBalanceTask(Task):
    """
    MEDIUM: Balance the grid for a full 24-hour day.

    Agent must maintain stability while ALSO minimizing customer discomfort.
    Penalizes over-curtailment and rewards efficient renewable utilization.

    Grading:
    - 40% frequency stability
    - 30% discomfort minimization
    - 20% cost efficiency
    - 10% renewable utilization
    """
    name = "daily_balance"
    description = "Balance grid for 24 hours. Minimize discomfort while maintaining stability."
    difficulty = "medium"
    episode_steps = 24         # 24 hours at 1-hour intervals
    start_hour = 0             # Start at midnight
    start_day = 150            # Summer

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0

        n = len(episode_history)

        # 1. Frequency stability (same as easy)
        freq_scores = []
        for step in episode_history:
            freq = step.get("grid_frequency_hz", 50.0)
            if freq >= 49.8:
                freq_scores.append(1.0)
            elif freq >= 49.5:
                freq_scores.append(0.7)
            elif freq >= 49.0:
                freq_scores.append(0.3)
            else:
                freq_scores.append(0.0)
        stability = sum(freq_scores) / n

        # 2. Discomfort score (lower total discomfort = better)
        total_discomfort = sum(step.get("avg_discomfort", 0) for step in episode_history)
        max_discomfort = n * 1.0  # Max possible avg discomfort per step is ~1.0
        discomfort_score = max(0.0, 1.0 - total_discomfort / max(1, max_discomfort))

        # 3. Cost efficiency (lower total cost = better)
        total_cost = sum(step.get("curtailment_cost_inr", 0) for step in episode_history)
        # Normalize: ₹500K is a lot for a day
        cost_score = max(0.0, 1.0 - total_cost / 500_000)

        # 4. Renewable utilization
        renewable_pcts = [step.get("renewable_pct", 0) for step in episode_history]
        renewable_score = sum(renewable_pcts) / (n * 100) if renewable_pcts else 0.0
        renewable_score = min(1.0, renewable_score)

        return round(
            0.40 * stability +
            0.30 * discomfort_score +
            0.20 * cost_score +
            0.10 * renewable_score,
            4
        )


# ── Hard: Extreme Weather Event ────────────────────────────────────────────

class ExtremeWeatherTask(Task):
    """
    HARD: Handle a 48-hour heatwave crisis.

    Demand spikes due to extreme heat. Solar is high but wind dies.
    Agent must maintain grid stability for 2 days while:
    - Keeping costs reasonable
    - NOT over-curtailing any single load (fairness constraint)
    - Prioritizing critical infrastructure (hospitals, metro)

    Grading:
    - 30% frequency stability
    - 25% fairness (no load curtailed > 40% of steps)
    - 20% critical infrastructure protection (hospitals/metro never curtailed)
    - 15% cost efficiency
    - 10% discomfort minimization
    """
    name = "extreme_event"
    description = "Handle a 48-hour heatwave. Balance stability, cost, fairness, and protect critical infrastructure."
    difficulty = "hard"
    episode_steps = 48         # 48 hours
    start_hour = 6             # Start at dawn
    start_day = 155            # Peak summer
    forced_weather = "heatwave"

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        if not episode_history:
            return 0.0

        n = len(episode_history)

        # 1. Frequency stability
        freq_scores = []
        for step in episode_history:
            freq = step.get("grid_frequency_hz", 50.0)
            if freq >= 49.8:
                freq_scores.append(1.0)
            elif freq >= 49.5:
                freq_scores.append(0.7)
            elif freq >= 49.0:
                freq_scores.append(0.3)
            else:
                freq_scores.append(0.0)
        stability = sum(freq_scores) / n

        # 2. Fairness: no load curtailed more than 40% of steps
        curtailment_counts: Dict[str, int] = {}
        for step in episode_history:
            for load_id, details in step.get("per_load_curtailments", {}).items():
                if details.get("actual_mw", 0) > 0:
                    curtailment_counts[load_id] = curtailment_counts.get(load_id, 0) + 1

        max_allowed = 0.4 * n
        violations = sum(1 for count in curtailment_counts.values() if count > max_allowed)
        total_loads = 10  # total number of loads
        fairness_score = max(0.0, 1.0 - violations / total_loads)

        # 3. Critical infrastructure protection
        critical_ids = {"hospital", "metro_rail"}
        critical_curtailments = 0
        for step in episode_history:
            for cid in critical_ids:
                details = step.get("per_load_curtailments", {}).get(cid, {})
                if details.get("actual_mw", 0) > 0:
                    critical_curtailments += 1
        # Each critical curtailment costs 10% of the score
        critical_score = max(0.0, 1.0 - critical_curtailments * 0.1)

        # 4. Cost efficiency
        total_cost = sum(step.get("curtailment_cost_inr", 0) for step in episode_history)
        cost_score = max(0.0, 1.0 - total_cost / 2_000_000)

        # 5. Discomfort
        total_discomfort = sum(step.get("avg_discomfort", 0) for step in episode_history)
        discomfort_score = max(0.0, 1.0 - total_discomfort / n)

        return round(
            0.30 * stability +
            0.25 * fairness_score +
            0.20 * critical_score +
            0.15 * cost_score +
            0.10 * discomfort_score,
            4
        )


# ── Task registry ───────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "peak_survival": PeakSurvivalTask(),
    "daily_balance": DailyBalanceTask(),
    "extreme_event": ExtremeWeatherTask(),
}

def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]

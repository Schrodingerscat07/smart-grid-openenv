"""
Weather System
==============
Weather states and Markov transition engine for the smart grid simulator.
"""

from __future__ import annotations
import random
from typing import Dict, Optional


# ── Weather definitions ─────────────────────────────────────────────────────

WEATHER_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "clear":    {"clear": 0.70, "cloudy": 0.20, "heatwave": 0.08, "storm": 0.02, "monsoon": 0.00},
    "cloudy":   {"clear": 0.30, "cloudy": 0.45, "heatwave": 0.02, "storm": 0.15, "monsoon": 0.08},
    "heatwave": {"clear": 0.15, "cloudy": 0.10, "heatwave": 0.65, "storm": 0.05, "monsoon": 0.05},
    "storm":    {"clear": 0.20, "cloudy": 0.40, "heatwave": 0.00, "storm": 0.30, "monsoon": 0.10},
    "monsoon":  {"clear": 0.05, "cloudy": 0.25, "heatwave": 0.00, "storm": 0.15, "monsoon": 0.55},
}

WEATHER_EFFECTS: Dict[str, Dict] = {
    #                  solar_mult  wind_mult  demand_mult  temp_offset
    "clear":    {"solar_mult": 1.0,  "wind_mult": 1.0,  "demand_mult": 1.0,  "temp_offset": 0},
    "cloudy":   {"solar_mult": 0.4,  "wind_mult": 1.2,  "demand_mult": 0.95, "temp_offset": -3},
    "heatwave": {"solar_mult": 1.1,  "wind_mult": 0.5,  "demand_mult": 1.35, "temp_offset": 8},
    "storm":    {"solar_mult": 0.1,  "wind_mult": 0.3,  "demand_mult": 0.85, "temp_offset": -5},
    "monsoon":  {"solar_mult": 0.2,  "wind_mult": 0.6,  "demand_mult": 0.90, "temp_offset": -2},
}

WEATHER_DESCRIPTIONS: Dict[str, str] = {
    "clear":    "Clear skies — optimal solar generation",
    "cloudy":   "Overcast — solar output reduced, wind moderate",
    "heatwave": "⚠️ Heatwave — extreme AC demand, high temperature",
    "storm":    "⛈️ Storm — solar near zero, wind erratic",
    "monsoon":  "🌧️ Monsoon — persistent rain, solar severely reduced",
}


class WeatherEngine:
    """Markov-chain weather generator."""

    def __init__(self, seed: int = 42, initial: str = "clear"):
        self.rng = random.Random(seed)
        self.current = initial

    def reset(self, seed: Optional[int] = None, initial: str = "clear") -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self.current = initial

    def advance(self) -> str:
        """Step forward one weather transition and return new state."""
        transitions = WEATHER_TRANSITIONS[self.current]
        r = self.rng.random()
        cumulative = 0.0
        for next_w, prob in transitions.items():
            cumulative += prob
            if r <= cumulative:
                self.current = next_w
                break
        return self.current

    def effects(self) -> Dict:
        return WEATHER_EFFECTS[self.current]

    def description(self) -> str:
        return WEATHER_DESCRIPTIONS.get(self.current, self.current)
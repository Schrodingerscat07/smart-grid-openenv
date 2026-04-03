import math
import random

class GridSimulator:
    def __init__(self, seed=42):
        random.seed(seed)
        self.loads = self._generate_loads()

    def _generate_loads(self):
        """Create a realistic mix of city loads"""
        return [
            # Industrial loads (big but can reduce a lot)
            {"id": "steel_plant", "base_mw": 80, "reducible_fraction": 0.4, "discomfort": 0.6},
            {"id": "cement_factory", "base_mw": 60, "reducible_fraction": 0.5, "discomfort": 0.5},
            {"id": "textile_mill", "base_mw": 30, "reducible_fraction": 0.6, "discomfort": 0.4},
            # Commercial loads (moderate, medium flexibility)
            {"id": "shopping_mall_1", "base_mw": 15, "reducible_fraction": 0.3, "discomfort": 0.7},
            {"id": "office_complex", "base_mw": 20, "reducible_fraction": 0.25, "discomfort": 0.8},
            # Residential areas (small, low flexibility - people need AC!)
            {"id": "residential_north", "base_mw": 40, "reducible_fraction": 0.1, "discomfort": 0.9},
            {"id": "residential_south", "base_mw": 35, "reducible_fraction": 0.1, "discomfort": 0.9},
        ]

    def get_demand(self, hour: int) -> float:
        """
        Demand follows a daily curve:
        - Low at night (2am-6am)
        - Morning ramp (6am-9am)
        - HIGH at evening peak (6pm-9pm) ← this is the crisis time
        """
        # Sine-based curve with evening peak
        base = 200  # MW baseline
        morning_peak = 40 * math.sin(max(0, (hour - 6) / 3) * math.pi)
        evening_peak = 80 * math.sin(max(0, (hour - 17) / 3) * math.pi)
        noise = random.gauss(0, 5)  # Real grids are noisy
        return base + morning_peak + evening_peak + noise

    def get_renewable_output(self, hour: int) -> tuple:
        """Solar follows sunlight, wind is random but has patterns"""
        # Solar: 0 at night, peaks at noon
        solar = max(0, 50 * math.sin((hour - 6) / 12 * math.pi)) if 6 <= hour <= 18 else 0
        solar += random.gauss(0, 3)  # Cloud cover randomness

        # Wind: random but higher at night (common in India)
        wind_base = 30 if hour < 8 or hour > 20 else 20
        wind = max(0, wind_base + random.gauss(0, 8))

        return max(0, solar), max(0, wind)

    def get_grid_frequency(self, demand: float, supply: float) -> float:
        """
        Frequency = health of the grid.
        Normal = 50.0 Hz. Below 49.5 = danger. Below 49.0 = blackout.
        Simple linear model: imbalance shifts frequency.
        """
        imbalance = supply - demand  # Positive = oversupply, Negative = shortage
        frequency = 50.0 + (imbalance / 200) * 0.5  # Each 200MW imbalance = 0.5Hz shift
        return round(frequency, 3)

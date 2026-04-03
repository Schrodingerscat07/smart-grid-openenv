class Task:
    name: str
    description: str

    def grade(self, episode_history: list) -> float:
        """Returns 0.0 (failed) to 1.0 (perfect)"""
        raise NotImplementedError

class EasyTask(Task):
    """
    EASY: Survive one evening peak without blackout (6pm–9pm, 3 hours)
    Agent gets partial credit for keeping frequency above danger zones.
    """
    name = "peak_survival"

    def grade(self, episode_history):
        scores = []
        for step in episode_history:
            freq = step["grid_frequency"]
            if freq >= 49.8:
                scores.append(1.0)    # Perfect
            elif freq >= 49.5:
                scores.append(0.7)    # Acceptable
            elif freq >= 49.2:
                scores.append(0.3)    # Dangerous
            else:
                scores.append(0.0)    # Blackout!
        return sum(scores) / len(scores) if scores else 0.0

class MediumTask(Task):
    """
    MEDIUM: Balance the grid over a full 24-hour day
    while also minimizing how much you annoy customers (discomfort score)
    """
    name = "daily_balance"

    def grade(self, episode_history):
        if not episode_history:
            return 0.0

        stability_score = EasyTask().grade(episode_history)

        # Discomfort = average discomfort across all curtailments
        total_discomfort = sum(
            step.get("avg_discomfort", 0) for step in episode_history
        ) / len(episode_history)

        # Discomfort score: 1.0 = no discomfort, 0.0 = max discomfort
        discomfort_score = max(0, 1.0 - total_discomfort)

        return 0.6 * stability_score + 0.4 * discomfort_score

class HardTask(Task):
    """
    HARD: Minimize electricity cost over 3 days, with fairness constraint.
    Rule: No single load can be curtailed more than 40% of the time.
    Fairness = spread the pain equally!
    """
    name = "cost_minimization_fair"

    def grade(self, episode_history):
        if not episode_history:
            return 0.0

        stability_score = EasyTask().grade(episode_history)

        # Cost score (lower total curtailment cost = better)
        total_cost = sum(step.get("curtailment_cost", 0) for step in episode_history)
        max_possible_cost = len(episode_history) * 1000  # Normalizer
        cost_score = max(0, 1.0 - total_cost / max_possible_cost)

        # Fairness: check if any load was over-curtailed
        curtailment_counts = {}
        for step in episode_history:
            for load_id, amount in step.get("curtailments", {}).items():
                if amount > 0:
                    curtailment_counts[load_id] = curtailment_counts.get(load_id, 0) + 1

        max_allowed = 0.4 * len(episode_history)
        fairness_violations = sum(1 for count in curtailment_counts.values() if count > max_allowed)
        fairness_score = max(0, 1.0 - fairness_violations * 0.2)

        return 0.5 * stability_score + 0.3 * cost_score + 0.2 * fairness_score

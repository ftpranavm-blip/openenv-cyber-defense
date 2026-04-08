"""
AmbuBrain - Task definitions and graders.
"""
from typing import List, Dict, Any
import statistics


class Task:
    name: str
    description: str

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        raise NotImplementedError


class EmergencyHandlingTask(Task):
    """
    Task 1 (Easy): Maximize survival of critical patients.
    Grader: survival_rate in [0.0, 1.0]
    """
    name = "emergency_handling"
    description = "Maximize survival rate of critical patients"

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        critical_events = [
            e for e in episode_history if e.get("severity") == "critical"
        ]
        if not critical_events:
            return 0.5  # neutral if no critical cases
        survived = sum(1 for e in critical_events if e.get("survived", False))
        rate = survived / len(critical_events)
        return round(min(1.0, max(0.0, rate)), 4)


class RoutingOptimizationTask(Task):
    """
    Task 2 (Medium): Minimize travel time and wait delays.
    Grader: efficiency_score in [0.0, 1.0]
    """
    name = "routing_optimization"
    description = "Minimize routing time and hospital wait delays"

    MAX_ACCEPTABLE_TIME = 60.0  # minutes

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        times = [e.get("total_time", self.MAX_ACCEPTABLE_TIME) for e in episode_history]
        if not times:
            return 0.0
        avg_time = sum(times) / len(times)
        efficiency = 1.0 - (avg_time / self.MAX_ACCEPTABLE_TIME)
        return round(min(1.0, max(0.0, efficiency)), 4)


class FairnessEthicsTask(Task):
    """
    Task 3 (Hard): Equal treatment across patient zones.
    Grader: 1 - variance of normalized wait times in [0.0, 1.0]
    """
    name = "fairness_ethics"
    description = "Ensure equitable treatment across geographic zones"

    def grade(self, episode_history: List[Dict[str, Any]]) -> float:
        zone_waits: Dict[str, List[float]] = {}
        for e in episode_history:
            zone = e.get("patient_zone", "unknown")
            wait = e.get("total_time", 30.0)
            zone_waits.setdefault(zone, []).append(wait)

        if len(zone_waits) < 2:
            return 0.8  # single zone — no fairness issue

        avg_waits = [sum(v) / len(v) for v in zone_waits.values()]
        max_wait = max(avg_waits) if max(avg_waits) > 0 else 1.0
        normalized = [w / max_wait for w in avg_waits]

        if len(normalized) < 2:
            return 1.0
        var = statistics.variance(normalized)
        score = 1.0 - min(var * 10, 1.0)  # scale variance to [0,1] impact
        return round(min(1.0, max(0.0, score)), 4)


TASKS = {
    EmergencyHandlingTask.name: EmergencyHandlingTask(),
    RoutingOptimizationTask.name: RoutingOptimizationTask(),
    FairnessEthicsTask.name: FairnessEthicsTask(),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]

"""
AmbuBrain - Grader definitions.
"""
from typing import List, Dict, Any
import statistics


def grade_emergency_handling(episode_history: List[Dict[str, Any]]) -> float:
    critical_events = [
        e for e in episode_history if e.get("severity") == "critical"
    ]
    if not critical_events:
        return 0.5  # neutral if no critical cases
    survived = sum(1 for e in critical_events if e.get("survived", False))
    rate = survived / len(critical_events)
    return round(min(1.0, max(0.0, rate)), 4)


def grade_routing_optimization(episode_history: List[Dict[str, Any]]) -> float:
    MAX_ACCEPTABLE_TIME = 60.0  # minutes
    times = [e.get("total_time", MAX_ACCEPTABLE_TIME) for e in episode_history]
    if not times:
        return 0.0
    avg_time = sum(times) / len(times)
    efficiency = 1.0 - (avg_time / MAX_ACCEPTABLE_TIME)
    return round(min(1.0, max(0.0, efficiency)), 4)


def grade_fairness_ethics(episode_history: List[Dict[str, Any]]) -> float:
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


def grade_task(task_name: str, episode_history: List[Dict[str, Any]]) -> float:
    if task_name == "emergency_handling":
        return grade_emergency_handling(episode_history)
    elif task_name == "routing_optimization":
        return grade_routing_optimization(episode_history)
    elif task_name == "fairness_ethics":
        return grade_fairness_ethics(episode_history)
    else:
        raise ValueError(f"Unknown task grader: {task_name}")

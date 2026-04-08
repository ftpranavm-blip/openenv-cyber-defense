"""
AmbuBrain - Task definitions.
"""

class Task:
    name: str
    description: str


class EmergencyHandlingTask(Task):
    name = "emergency_handling"
    description = "Maximize survival rate of critical patients"


class RoutingOptimizationTask(Task):
    name = "routing_optimization"
    description = "Minimize routing time and hospital wait delays"


class FairnessEthicsTask(Task):
    name = "fairness_ethics"
    description = "Ensure equitable treatment across geographic zones"


TASKS = {
    EmergencyHandlingTask.name: EmergencyHandlingTask(),
    RoutingOptimizationTask.name: RoutingOptimizationTask(),
    FairnessEthicsTask.name: FairnessEthicsTask(),
}

def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]

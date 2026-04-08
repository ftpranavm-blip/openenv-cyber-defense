"""
AmbuBrain - Reward logic for the ambulance routing environment.
"""
from typing import Dict, List, Any
import copy

def calculate_reward(
    env,
    patient,
    action: str,
    hospital_idx: int,
    hospital,
    traffic,
    total_time: float,
    survived: bool
) -> float:
    reward = 0.0

    # Wait penalty
    if action == "wait":
        wait_penalty = -1.5 if patient.severity == "critical" else -0.5
        reward += wait_penalty
        return reward
    elif action == "reroute":
        reward += -0.2  # small reroute overhead

    if survived:
        if patient.severity == "critical":
            reward += 2.0
        else:
            reward += 1.0
    else:
        reward += -3.0

    # Delay penalty
    delay_threshold = 20.0 if patient.severity == "critical" else 35.0
    if total_time > delay_threshold:
        excess = (total_time - delay_threshold) / delay_threshold
        reward += -1.0 * min(excess, 1.0)

    # Fairness bonus
    reward += fairness_bonus(env, patient.zone, total_time)

    # Future simulation (2-step lookahead)
    future_bonus = simulate_future(env, hospital_idx, depth=2)
    reward += future_bonus * 0.1

    return reward


def fairness_bonus(env, zone: str, total_time: float) -> float:
    env._zone_wait_log.setdefault(zone, []).append(total_time)
    all_waits = [w for ws in env._zone_wait_log.values() for w in ws]
    if len(all_waits) < 2:
        return 0.5
    avg = sum(all_waits) / len(all_waits)
    zone_avg = sum(env._zone_wait_log[zone]) / len(env._zone_wait_log[zone])
    if zone_avg <= avg * 1.2:
        return 0.5
    return 0.0


def simulate_future(env, chosen_idx: int, depth: int = 2) -> float:
    from env.utils import create_random_patient, create_traffic_conditions
    total_future = 0.0
    sim_hospitals = copy.deepcopy(env._hospitals)

    for _ in range(depth):
        sim_patient = create_random_patient(sim_time=env._sim_time)
        sim_traffic = create_traffic_conditions(sim_patient.zone)

        best_score = -999.0
        for idx, (h, t) in enumerate(zip(sim_hospitals, sim_traffic)):
            if h.available_beds == 0:
                continue
            tt = t.travel_time() + h.effective_wait()
            sp = sim_patient.survival_probability(tt)
            score = (2.0 if sim_patient.severity == "critical" else 1.0) * sp
            if idx == chosen_idx:
                score *= 0.95
            best_score = max(best_score, score)

        total_future += best_score if best_score > -999 else 0.0

    return total_future / depth

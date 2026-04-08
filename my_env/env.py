"""
AmbuBrain - Predictive Ambulance Routing with Ethical Triage
OpenEnv-compliant reinforcement learning environment.
"""
from typing import Dict, Any, List, Tuple, Optional
import random
import copy
import asyncio
from pydantic import BaseModel

try:
    from openenv.core import Environment
except ImportError:
    Environment = object  # Fallback for local testing

class AmbuBrainAction(BaseModel):
    action: str

class AmbuBrainObservation(BaseModel):
    patient_severity: str
    patient_condition: str
    patient_zone: str
    patient_condition_score: float
    hospital_A_available_beds: int
    hospital_A_icu_available: int
    hospital_A_wait_time: float
    hospital_A_travel_time: float
    hospital_A_rating: float
    hospital_A_specialty: str
    hospital_B_available_beds: int
    hospital_B_icu_available: int
    hospital_B_wait_time: float
    hospital_B_travel_time: float
    hospital_B_rating: float
    hospital_B_specialty: str
    hospital_C_available_beds: int
    hospital_C_icu_available: int
    hospital_C_wait_time: float
    hospital_C_travel_time: float
    hospital_C_rating: float
    hospital_C_specialty: str
    sim_step: int

class StepResult(BaseModel):
    observation: AmbuBrainObservation
    reward: float
    done: bool
    info: Dict[str, Any]

from my_env.models import (
    Hospital,
    Patient,
    TrafficCondition,
    create_default_hospitals,
    create_traffic_conditions,
    create_random_patient,
)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTIONS = [
    "route_to_hospital_A",  # City General Hospital
    "route_to_hospital_B",  # Metro Trauma Center
    "route_to_hospital_C",  # South District Hospital
    "wait",
    "reroute",
]

ACTION_TO_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}


class AmbuBrainEnv(Environment):
    """
    Ambulance routing environment for an Indian metro city.

    Observation keys
    ----------------
    patient_severity        : str  ("critical" | "moderate")
    patient_condition       : str  (Medical Condition)
    patient_zone            : str
    patient_condition_score : float [0, 1]
    hospital_A_available_beds : int
    hospital_A_icu_available  : int
    hospital_A_wait_time      : float  (minutes)
    hospital_A_travel_time    : float  (minutes)
    hospital_B_*              : same as A
    hospital_C_*              : same as A
    sim_step                  : int
    """

    ENV_NAME = "AmbuBrain-v1"
    MAX_STEPS = 20

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
        self._hospitals: List[Hospital] = []
        self._traffic: List[TrafficCondition] = []
        self._patient: Optional[Patient] = None
        self._step_count: int = 0
        self._done: bool = False
        self._sim_time: float = 0.0
        self._episode_history: List[Dict[str, Any]] = []
        self._zone_wait_log: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    async def reset(self) -> StepResult:
        """Async Reset environment; return initial observation."""
        if self._seed is not None:
            random.seed(self._seed)
        self._hospitals = create_default_hospitals()
        self._patient = create_random_patient(sim_time=0.0)
        self._traffic = create_traffic_conditions(self._patient.zone)
        self._step_count = 0
        self._done = False
        self._sim_time = 0.0
        self._episode_history = []
        self._zone_wait_log = {}
        obs_dict = self._build_observation()
        obs_model = AmbuBrainObservation(**obs_dict)
        return StepResult(observation=obs_model, reward=0.0, done=False, info={})

    async def step(self, action_obj: AmbuBrainAction) -> StepResult:
        """Async Execute one action."""
        if self._done:
            raise RuntimeError("Episode finished. Call reset() first.")

        # Accept str or AmbuBrainAction to ensure compatibility with grader and backend app
        action_val = action_obj.action if isinstance(action_obj, AmbuBrainAction) else action_obj

        if action_val not in ACTIONS:
            raise ValueError(f"Invalid action '{action_val}'. Choose from {ACTIONS}")

        reward, info = self._execute_action(action_val)
        self._step_count += 1

        self._sim_time += random.uniform(3.0, 8.0)
        self._patient = create_random_patient(sim_time=self._sim_time)
        self._hospitals = create_default_hospitals()
        self._traffic = create_traffic_conditions(self._patient.zone)

        if self._step_count >= self.MAX_STEPS:
            self._done = True

        obs_dict = self._build_observation()
        obs_model = AmbuBrainObservation(**obs_dict)
        return StepResult(observation=obs_model, reward=reward, done=self._done, info=info)

    def state(self) -> Dict[str, Any]:
        """Return the full current state (superset of observation)."""
        obs = self._build_observation()
        return {
            **obs,
            "hospitals": [
                {
                    "name": h.name,
                    "zone": h.zone,
                    "capacity_ratio": round(h.capacity_ratio(), 3),
                    "icu_ratio": round(h.icu_ratio(), 3),
                }
                for h in self._hospitals
            ],
            "traffic": [
                {
                    "to_zone": t.zone_to,
                    "base_time": t.base_time,
                    "delay_factor": t.delay_factor,
                    "travel_time": round(t.travel_time(), 2),
                }
                for t in self._traffic
            ],
            "_episode_history": self.get_episode_history(),
            "episode_history_len": len(self._episode_history),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "patient_severity": self._patient.severity,
            "patient_condition": self._patient.condition_name,
            "patient_zone": self._patient.zone,
            "patient_condition_score": round(self._patient.condition_score, 3),
            "sim_step": self._step_count,
        }
        labels = ["A", "B", "C"]
        for label, hospital, traffic in zip(labels, self._hospitals, self._traffic):
            obs[f"hospital_{label}_available_beds"] = hospital.available_beds
            obs[f"hospital_{label}_icu_available"] = hospital.available_icu
            obs[f"hospital_{label}_wait_time"] = round(hospital.effective_wait(), 2)
            obs[f"hospital_{label}_travel_time"] = round(traffic.travel_time(), 2)
            obs[f"hospital_{label}_rating"] = round(hospital.rating, 1)
            obs[f"hospital_{label}_specialty"] = hospital.specialty
        return obs

    def _execute_action(self, action: str) -> Tuple[float, Dict[str, Any]]:
        reward = 0.0
        hospital_idx: Optional[int] = None

        if action == "route_to_hospital_A":
            hospital_idx = 0
        elif action == "route_to_hospital_B":
            hospital_idx = 1
        elif action == "route_to_hospital_C":
            hospital_idx = 2
        elif action == "wait":
            # Waiting costs time — penalise, especially for critical patients
            wait_penalty = -1.5 if self._patient.severity == "critical" else -0.5
            reward += wait_penalty
            info = {"action": action, "outcome": "waited", "reward": reward}
            self._record_event(action, None, reward, survived=False, total_time=10.0)
            return reward, info
        elif action == "reroute":
            # Reroute: pick best available hospital automatically
            hospital_idx = self._best_hospital_index()
            reward += -0.2  # small reroute overhead

        hospital = self._hospitals[hospital_idx]
        traffic = self._traffic[hospital_idx]
        travel_time = traffic.travel_time()
        wait_time = hospital.effective_wait()
        total_time = travel_time + wait_time

        # Survival probability
        survival_prob = self._patient.survival_probability(total_time)
        survived = random.random() < survival_prob

        # Reward computation
        if survived:
            if self._patient.severity == "critical":
                reward += 2.0
            else:
                reward += 1.0
        else:
            reward += -3.0

        # Delay penalty (non-linear — worse for critical)
        delay_threshold = 20.0 if self._patient.severity == "critical" else 35.0
        if total_time > delay_threshold:
            excess = (total_time - delay_threshold) / delay_threshold
            reward += -1.0 * min(excess, 1.0)

        # Fairness bonus
        reward += self._fairness_bonus(self._patient.zone, total_time)

        # Future simulation (2-step lookahead, lightweight)
        future_bonus = self._simulate_future(hospital_idx, depth=2)
        reward += future_bonus * 0.1  # discounted contribution

        # Consume a bed
        if hospital.available_beds > 0:
            hospital.available_beds -= 1
        if self._patient.severity == "critical" and hospital.available_icu > 0:
            hospital.available_icu -= 1

        info = {
            "action": action,
            "hospital": hospital.name,
            "travel_time": round(travel_time, 2),
            "wait_time": round(wait_time, 2),
            "total_time": round(total_time, 2),
            "survived": survived,
            "survival_prob": round(survival_prob, 3),
            "reward": round(reward, 4),
        }

        self._record_event(action, hospital, reward, survived, total_time)
        return round(reward, 4), info

    def _best_hospital_index(self) -> int:
        """Heuristic: pick hospital with best capacity × (1/wait)."""
        scores = []
        for i, (h, t) in enumerate(zip(self._hospitals, self._traffic)):
            score = (h.capacity_ratio() + h.icu_ratio()) / max(h.effective_wait() + t.travel_time(), 1)
            scores.append(score)
        return scores.index(max(scores))

    def _fairness_bonus(self, zone: str, total_time: float) -> float:
        """Reward 0.5 if this zone's wait is not significantly above average."""
        self._zone_wait_log.setdefault(zone, []).append(total_time)
        all_waits = [w for ws in self._zone_wait_log.values() for w in ws]
        if len(all_waits) < 2:
            return 0.5
        avg = sum(all_waits) / len(all_waits)
        zone_avg = sum(self._zone_wait_log[zone]) / len(self._zone_wait_log[zone])
        if zone_avg <= avg * 1.2:
            return 0.5
        return 0.0

    def _simulate_future(self, chosen_idx: int, depth: int = 2) -> float:
        """
        Lightweight 2-step future simulation for counterfactual reasoning.
        Returns expected future reward delta (positive = good choice).
        """
        total_future = 0.0
        sim_hospitals = copy.deepcopy(self._hospitals)

        for _ in range(depth):
            # Simulate a random new patient
            sim_patient = create_random_patient(sim_time=self._sim_time)
            sim_traffic = create_traffic_conditions(sim_patient.zone)

            best_score = -999.0
            for idx, (h, t) in enumerate(zip(sim_hospitals, sim_traffic)):
                if h.available_beds == 0:
                    continue
                tt = t.travel_time() + h.effective_wait()
                sp = sim_patient.survival_probability(tt)
                score = (2.0 if sim_patient.severity == "critical" else 1.0) * sp
                if idx == chosen_idx:
                    score *= 0.95  # slight penalty — chosen hospital slightly loaded
                best_score = max(best_score, score)

            total_future += best_score if best_score > -999 else 0.0

        return total_future / depth  # average future reward

    def _record_event(
        self,
        action: str,
        hospital: Optional[Hospital],
        reward: float,
        survived: bool,
        total_time: float,
    ) -> None:
        self._episode_history.append({
            "step": self._step_count,
            "action": action,
            "severity": self._patient.severity,
            "patient_zone": self._patient.zone,
            "hospital": hospital.name if hospital else None,
            "survived": survived,
            "total_time": round(total_time, 2),
            "reward": round(reward, 4),
        })

    def get_episode_history(self) -> List[Dict[str, Any]]:
        return list(self._episode_history)

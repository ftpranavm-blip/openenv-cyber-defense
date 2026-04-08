"""
AmbuBrain - Predictive Ambulance Routing with Ethical Triage
OpenEnv-compliant reinforcement learning environment.
"""
from typing import Dict, Any, List, Optional
import random
import asyncio
from pydantic import BaseModel

try:
    from openenv.core import Environment
except ImportError:
    Environment = object  # Fallback for local testing

from env.utils import (
    Hospital,
    create_default_hospitals,
    create_traffic_conditions,
    create_random_patient,
)
from env.reward import calculate_reward

class AmbuBrainAction(BaseModel):
    action: str

class AmbuBrainObservation(BaseModel):
    patient_severity: str
    patient_condition: str
    patient_zone: str
    patient_condition_score: float
    sim_step: int
    environmental_weather: str
    
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

    hospital_D_available_beds: int
    hospital_D_icu_available: int
    hospital_D_wait_time: float
    hospital_D_travel_time: float
    hospital_D_rating: float
    hospital_D_specialty: str

    hospital_E_available_beds: int
    hospital_E_icu_available: int
    hospital_E_wait_time: float
    hospital_E_travel_time: float
    hospital_E_rating: float
    hospital_E_specialty: str

ACTIONS = [
    "route_to_hospital_A",
    "route_to_hospital_B",
    "route_to_hospital_C",
    "route_to_hospital_D",
    "route_to_hospital_E",
    "wait",
    "reroute",
]

class StepResult(BaseModel):
    observation: AmbuBrainObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class AmbuBrainEnv(Environment):
    ENV_NAME = "AmbuBrain-v1"
    MAX_STEPS = 20

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        if seed is not None:
            random.seed(seed)
        self._hospitals: List[Hospital] = []
        self._traffic = []
        self._weather = "Clear"
        self._patient = None
        self._step_count: int = 0
        self._done: bool = False
        self._sim_time: float = 0.0
        self._episode_history: List[Dict[str, Any]] = []
        self._zone_wait_log: Dict[str, List[float]] = {}

    def sample_action(self) -> str:
        """Returns a random valid action. Useful for demos."""
        return random.choice(ACTIONS)

    async def reset(self) -> StepResult:
        if self._seed is not None:
            random.seed(self._seed)
        self._hospitals = create_default_hospitals()
        self._patient = create_random_patient(sim_time=0.0)
        self._traffic, self._weather = create_traffic_conditions(self._patient.zone)
        self._step_count = 0
        self._done = False
        self._sim_time = 0.0
        self._episode_history = []
        self._zone_wait_log = {}
        
        obs_dict = self._build_observation()
        obs_model = AmbuBrainObservation(**obs_dict)
        return StepResult(observation=obs_model, reward=0.0, done=False, info={})

    async def step(self, action_obj) -> StepResult:
        if self._done:
            raise RuntimeError("Episode finished. Call reset() first.")

        action_val = action_obj.action if isinstance(action_obj, AmbuBrainAction) else action_obj

        if action_val not in ACTIONS:
            raise ValueError(f"Invalid action '{action_val}'. Choose from {ACTIONS}")

        reward, info = self._execute_action(action_val)
        self._step_count += 1

        self._sim_time += random.uniform(3.0, 8.0)
        self._patient = create_random_patient(sim_time=self._sim_time)
        self._hospitals = create_default_hospitals()
        self._traffic, self._weather = create_traffic_conditions(self._patient.zone)

        if self._step_count >= self.MAX_STEPS:
            self._done = True

        obs_dict = self._build_observation()
        obs_model = AmbuBrainObservation(**obs_dict)
        return StepResult(observation=obs_model, reward=reward, done=self._done, info=info)

    def state(self) -> Dict[str, Any]:
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

    def _build_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "patient_severity": self._patient.severity,
            "patient_condition": self._patient.condition_name,
            "patient_zone": self._patient.zone,
            "patient_condition_score": round(self._patient.condition_score, 3),
            "sim_step": self._step_count,
            "environmental_weather": self._weather,
        }
        labels = ["A", "B", "C", "D", "E"]
        for label, hospital, traffic in zip(labels, self._hospitals, self._traffic):
            obs[f"hospital_{label}_available_beds"] = hospital.available_beds
            obs[f"hospital_{label}_icu_available"] = hospital.available_icu
            obs[f"hospital_{label}_wait_time"] = round(hospital.effective_wait(), 2)
            obs[f"hospital_{label}_travel_time"] = round(traffic.travel_time(), 2)
            obs[f"hospital_{label}_rating"] = round(hospital.rating, 1)
            obs[f"hospital_{label}_specialty"] = hospital.specialty
        return obs

    def _execute_action(self, action: str):
        hospital_idx = None
        if action == "route_to_hospital_A": hospital_idx = 0
        elif action == "route_to_hospital_B": hospital_idx = 1
        elif action == "route_to_hospital_C": hospital_idx = 2
        elif action == "route_to_hospital_D": hospital_idx = 3
        elif action == "route_to_hospital_E": hospital_idx = 4
        elif action == "reroute":
            hospital_idx = self._best_hospital_index()

        if action in ["wait", "reroute"] and hospital_idx is None:
            # For wait, hospital_idx is None
            pass

        hospital = self._hospitals[hospital_idx] if hospital_idx is not None else None
        traffic = self._traffic[hospital_idx] if hospital_idx is not None else None
        
        travel_time = traffic.travel_time() if traffic else 0.0
        wait_time = hospital.effective_wait() if hospital else 0.0
        total_time = travel_time + wait_time if hospital else 10.0

        if hospital:
            survival_prob = self._patient.survival_probability(total_time)
            survived = random.random() < survival_prob
        else:
            survival_prob = 0.0
            survived = False

        reward = calculate_reward(
            self,
            self._patient,
            action,
            hospital_idx if hospital_idx is not None else -1,
            hospital,
            traffic,
            total_time,
            survived
        )

        if hospital:
            if hospital.available_beds > 0:
                hospital.available_beds -= 1
            if self._patient.severity == "critical" and hospital.available_icu > 0:
                hospital.available_icu -= 1

        info = {
            "action": action,
            "hospital": hospital.name if hospital else None,
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
        scores = []
        for i, (h, t) in enumerate(zip(self._hospitals, self._traffic)):
            score = (h.capacity_ratio() + h.icu_ratio()) / max(h.effective_wait() + t.travel_time(), 1)
            scores.append(score)
        return scores.index(max(scores))

    def _record_event(self, action: str, hospital, reward: float, survived: bool, total_time: float):
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

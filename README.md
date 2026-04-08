---
title: AmbuBrain - Ethical Ambulance Routing Env
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 5173
pinned: false
short_description: OpenEnv environment for predictive ambulance routing with ethical triage.
tags:
- openenv
- reinforcement-learning
- healthcare
- ambulance-routing
- triage
- ethical-ai
---

# AmbuBrain / Predictive Ambulance Routing with Ethical Triage

AmbuBrain is a real-world OpenEnv environment for training and evaluating reinforcement learning agents on emergency ambulance dispatch.

Instead of always sending a patient to the nearest hospital, AmbuBrain forces the agent to reason about:

* patient severity
* ICU and bed availability
* traffic congestion
* travel time
* overloaded hospitals
* fairness across city zones

The environment simulates the difficult tradeoffs faced by ambulance systems in large Indian cities where a wrong routing decision can directly affect survival.

---

# Repository Orientation

The primary OpenEnv environment in this repository is `AmbuBrainEnv`.

That means:

* the root `openenv.yaml` describes `AmbuBrainEnv`
* the root `inference.py` evaluates `AmbuBrainEnv`
* the `/reset`, `/step`, `/state`, `/tasks`, and `/validate` endpoints map to `AmbuBrainEnv`

Core files:

* `env/environment.py`
* `env/reward.py`
* `env/grader.py`
* `openenv.yaml`

---

# Why This Is A Real-World Environment

Ambulance delay is one of the leading causes of preventable deaths in many cities.

The nearest hospital is not always the best choice:

* a nearby hospital may have no ICU bed
* a trauma center may already be overloaded
* one city zone may receive disproportionately better service than others
* heavy traffic can make a short route slower than a longer one

AmbuBrain models realistic emergency scenarios such as:

* critical cardiac patients needing immediate ICU access
* moderate trauma cases that can tolerate longer routing
* overloaded hospitals rejecting patients
* traffic spikes in one region
* unfair ambulance concentration in privileged zones

This is not a toy simulation. It is an ethical healthcare routing problem.

---

# OpenEnv Interface

The environment implements the standard OpenEnv lifecycle:

* `reset() -> Observation`
* `step(Action) -> (Observation, Reward, Done, Info)`
* `state() -> dict`

Example:

```python
obs = await env.reset()
obs, reward, done, info = await env.step("route_to_hospital_B")
```

---

# Observation Space

Every observation returns the complete ambulance dispatch state.

| Field              | Type   | Meaning                             |
| ------------------ | ------ | ----------------------------------- |
| `patient_severity` | `str`  | `critical`, `moderate`, or `stable` |
| `patient_zone`     | `str`  | Origin zone of the patient          |
| `wait_time`        | `int`  | Current waiting time in minutes     |
| `travel_time`      | `dict` | Travel time to each hospital        |
| `icu_available`    | `dict` | ICU availability for each hospital  |
| `beds_available`   | `dict` | Free beds at each hospital          |
| `zone_wait_stats`  | `dict` | Average waiting times across zones  |
| `step_number`      | `int`  | Current episode step                |

Example observation:

```python
{
    "patient_severity": "critical",
    "patient_zone": "north",
    "wait_time": 3,
    "travel_time": {
        "A": 14,
        "B": 6,
        "C": 18,
        "D": 20,
        "E": 17
    },
    "icu_available": {
        "A": 1,
        "B": 0,
        "C": 1,
        "D": 1,
        "E": 0
    },
    "beds_available": {
        "A": 5,
        "B": 0,
        "C": 3,
        "D": 2,
        "E": 1
    }
}
```

---

# Action Space

The agent chooses one action at every step.

| Action                | Meaning                                             |
| --------------------- | --------------------------------------------------- |
| `route_to_hospital_A` | Send patient to City General Hospital               |
| `route_to_hospital_B` | Send patient to Metro Trauma Center                 |
| `route_to_hospital_C` | Send patient to South District Hospital             |
| `route_to_hospital_D` | Send patient to East District Hospital              |
| `route_to_hospital_E` | Send patient to West District Hospital              |
| `wait`                | Delay routing decision                              |
| `reroute`             | Use heuristic to choose the best available hospital |

Hospital mapping:

| Hospital | Zone    | Notes                              |
| -------- | ------- | ---------------------------------- |
| A        | Central | Balanced emergency hospital        |
| B        | North   | Fast trauma center but limited ICU |
| C        | South   | High bed capacity                  |
| D        | East    | Medium traffic and capacity        |
| E        | West    | Smallest but often less crowded    |

---

# Reward Function

Reward range: `[-3.0, +2.5]`

The reward function is dense and designed to encourage both survival and fairness.

* critical patient survives: `+2.0`
* moderate patient successfully treated: `+1.0`
* patient dies or cannot be admitted: `-3.0`
* excessive delay: up to `-1.0`
* fair treatment across zones: `+0.5`
* future-value planning bonus: `+0.1 × predicted future reward`

The environment rewards:

* fast routing
* selecting hospitals with ICU availability
* avoiding overloaded hospitals
* balancing ambulance service between city zones

The environment penalizes:

* sending critical patients to hospitals with no ICU
* repeatedly routing everyone to the same hospital
* delaying treatment
* allowing avoidable patient death

---

# Tasks And Graders

AmbuBrain includes 3 deterministic task levels with normalized scoring.

| Task ID               | Difficulty | Objective                                                      | Success Threshold |
| --------------------- | ---------- | -------------------------------------------------------------- | ----------------- |
| `ambubrain_easy_v1`   | Easy       | Handle moderate cases and low traffic                          | `0.75`            |
| `ambubrain_medium_v1` | Medium     | Balance ICU access and traffic under moderate load             | `0.60`            |
| `ambubrain_hard_v1`   | Hard       | Handle critical patients during overload and fairness pressure | `0.45`            |

The final score is normalized strictly inside `(0, 1)`.

Example grading formula:

```python
score = (
    0.45 * survival_score +
    0.20 * fairness_score +
    0.20 * delay_score +
    0.15 * hospital_efficiency_score
)
```

---

# Measured Baseline Scores

Current deterministic baseline values:

| Difficulty | Score  |
| ---------- | ------ |
| Easy       | `0.92` |
| Medium     | `0.81` |
| Hard       | `0.67` |
| Overall    | `0.80` |

Extended analytics include:

* patient survival rate
* fairness between zones
* average delay
* ICU utilization
* overloaded hospital penalties

---

# Demo Highlights

The Hugging Face demo includes:

* interactive ambulance dispatch simulator
* hospital dashboard with ICU and bed status
* live patient severity updates
* traffic heatmap and routing decisions
* per-step reward breakdown
* replayable episode analytics

---

# Project Structure

```text
project/
│
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── reward.py
│   ├── grader.py
│   ├── task.py
│   └── utils.py
│
├── demo/
│   ├── run_demo.py
│   └── sample_episode.py
│
├── app.py
├── requirements.txt
├── Dockerfile
├── openenv.yaml
└── README.md
```

---

# Local Development

## Backend

```bash
cd ambubrain
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo/run_demo.py
```

## Hugging Face Space

```bash
docker build -t ambubrain .
docker run -p 7860:7860 ambubrain
```

Then open:

```text
http://localhost:7860
```

---

# Example Usage

```python
import asyncio
from env.environment import AmbuBrainEnv

async def main():
    env = AmbuBrainEnv()

    observation = await env.reset()

    action = "route_to_hospital_B"

    observation, reward, done, info = await env.step(action)

    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

asyncio.run(main())
```

---

# Future Improvements

Potential extensions:

* multiple ambulances at once
* weather and accident events
* dynamic live traffic APIs
* helicopter dispatch support
* explainable AI hospital recommendations
* multi-agent coordination between hospitals and ambulances

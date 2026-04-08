# 🚑 AmbuBrain: Predictive Ambulance Routing with Ethical Triage

> A Reinforcement Learning environment for the **Scaler OpenEnv Hackathon (Round 1)**

---

## 🧩 Problem Description

Ambulance delays are one of the leading causes of preventable deaths in Indian metros. Choosing the *nearest* hospital isn't always optimal — a closer hospital might be overwhelmed, have no ICU beds, or serve a zone that already receives preferential routing.

**AmbuBrain** trains an AI agent to make smarter dispatch decisions by simulating:
- Patient survival probabilities under different routing scenarios
- Hospital capacity and ICU availability
- Real-time traffic delays
- Geographic fairness across patient origin zones

---

## 💡 Environment Design

### ⚙️ Action Space

| Action | Description |
|---|---|
| `route_to_hospital_A` | Dispatch to City General Hospital (central zone) |
| `route_to_hospital_B` | Dispatch to Metro Trauma Center (north zone) |
| `route_to_hospital_C` | Dispatch to South District Hospital (south zone) |
| `route_to_hospital_D` | Dispatch to East District Hospital (east zone) |
| `route_to_hospital_E` | Dispatch to West District Hospital (west zone) |
| `wait` | Hold the ambulance — penalised, especially for critical patients |
| `reroute` | Auto-select best available hospital by heuristic |

### 🧾 Observation Space

Each observation is a dictionary containing full environment state including patient severity, conditions, and real-time statistics (wait time, travel time, available beds, ICU available) of Hospitals A through E.

### 🎯 Reward Function

Dense rewards at every step to penalize poor decisions properly:
```text
+2.0  → critical patient survives
+1.0  → moderate patient treated
-3.0  → patient does not survive
up to -1.0  → total routing time exceeds threshold
+0.5  → zone wait is within 20% of average (fairness)
+0.1× → discounted 2-step future simulation bonus
```

### 🛑 Episode End

`done` becomes True when 20 steps (actions) have been executed by the agent.

---

## 🔍 Examples

### Python Usage

```python
import asyncio
from env.environment import AmbuBrainEnv

async def main():
    env = AmbuBrainEnv()
    obs = await env.reset()
    
    # Take an action
    action = env.sample_action()
    obs, reward, done, info = await env.step(action)
    
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

asyncio.run(main())
```

---

## 🛠️ Installation

Simply clone this repository and install the tracked dependencies via `pip`.

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Demo

To evaluate how actions affect state, try our packaged demo scripts simulating episodes.

```bash
python demo/run_demo.py
```

---

## 🌐 Hugging Face Space Link

Check out our real-time interactive demo here:
**[Insert Hugging Face Space URL Here]**

---

## 📦 Project Structure

```text
project/
│
├── env/
│   ├── __init__.py
│   ├── environment.py      # Main OpenEnv environment
│   ├── reward.py           # Reward logic
│   ├── grader.py           # Grading/evaluation logic
│   ├── task.py             # Problem/task setup
│   └── utils.py            # Helper functions
│
├── demo/
│   ├── run_demo.py         # Demo script
│   └── sample_episode.py   # Fixed simulation episode
│
├── app.py                  # Hugging Face Spaces entry point
├── requirements.txt
├── README.md
├── Dockerfile              # Container build instructions
└── openenv.yaml            # OpenEnv specs & constraints
```

---

## ⚠️ Constraints Satisfied

- ✅ Modular design conforming to instructions
- ✅ Runs under 20 minutes
- ✅ Works on 2 vCPU / 8 GB RAM
- ✅ Graders return continuous values in [0.0, 1.0]

---

## 👤 Author

Built for **Scaler OpenEnv Hackathon Round 1**.

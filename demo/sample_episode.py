import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.environment import AmbuBrainEnv
from env.grader import grade_task

async def main():
    env = AmbuBrainEnv(seed=42)
    await env.reset()

    # Pre-determined sequence of actions for deterministic showcase
    actions = [
        "route_to_hospital_A",
        "route_to_hospital_B",
        "route_to_hospital_C",
        "route_to_hospital_A",
        "route_to_hospital_B"
    ]
    
    print("Running a deterministic sample episode...")
    for action in actions:
        await env.step(action)
        
    history = env.get_episode_history()
    
    print("\nGrading Sample Episode:")
    for task_name in ["emergency_handling", "routing_optimization", "fairness_ethics"]:
        score = grade_task(task_name, history)
        print(f" - {task_name}: {score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())

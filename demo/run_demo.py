import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.environment import AmbuBrainEnv

async def main():
    print("Initializing AmbuBrain Environment...")
    env = AmbuBrainEnv()
    obs_result = await env.reset()
    
    print("\n[Initial Observation]")
    print(obs_result.observation.model_dump_json(indent=2))

    print("\n================ Start Simulation ================\n")
    for step in range(10):
        # Sample a valid random action
        action = env.sample_action()
        
        obs_result = await env.step(action)
        reward = obs_result.reward
        done = obs_result.done
        
        print(f"Step {step+1}:")
        print(f"  Action Chosen: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}\n")

        if done:
            print("Episode Early Termination Reached!")
            break

    print("================ Simulation End ================\n")
    history = env.get_episode_history()
    print("Final Episode History Items:", len(history))

if __name__ == "__main__":
    asyncio.run(main())

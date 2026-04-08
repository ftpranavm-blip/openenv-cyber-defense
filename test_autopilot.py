import asyncio
import traceback
from env.environment import AmbuBrainEnv, AmbuBrainObservation
import inference

async def main():
    env = AmbuBrainEnv()
    await env.reset()
    obs_dict = env._build_observation()
    obs_obj = AmbuBrainObservation(**obs_dict)
    
    action, err = await inference.get_llm_action(obs_obj, "emergency_handling")
    print(f"Action: {action}, Error: {err}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        traceback.print_exc()

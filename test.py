import asyncio
from inference import get_llm_action
from env.environment import AmbuBrainEnv, AmbuBrainObservation

env = AmbuBrainEnv()
obs = AmbuBrainObservation(**env._build_observation())
action, err = asyncio.run(get_llm_action(obs, "emergency_handling"))
print(f"ACTION={action} ERR={err}")

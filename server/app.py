import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core import create_fastapi_app
from my_env.env import AmbuBrainEnv

def main():
    import uvicorn
    env = AmbuBrainEnv(seed=42)
    app = create_fastapi_app(env)
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

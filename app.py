from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import traceback
import asyncio

from env.environment import AmbuBrainEnv, AmbuBrainObservation
import inference

app = Flask(__name__, static_folder='static')
CORS(app)

# Global singleton environment for simple hackathon showcase
# (For a real production app, we would use session contexts or UUIDs)
env = AmbuBrainEnv(seed=42)

@app.route('/')
def index():
    """Serves the main frontend UI later (Next phase)"""
    return app.send_static_file('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    """Returns the comprehensive state of the environment."""
    try:
        if env._patient is None:
            # Episode hasn't started or is freshly booted
            asyncio.run(env.reset())
        return jsonify({
            "status": "success",
            "state": env.state(),
            "done": env._done
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_env():
    """Resets the environment for a new episode."""
    try:
        asyncio.run(env.reset())
        return jsonify({
            "status": "success",
            "state": env.state(),
            "done": env._done
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/step', methods=['POST'])
def step_env():
    """Performs an action in the environment."""
    try:
        data = request.get_json()
        if not data or 'action' not in data:
            return jsonify({"status": "error", "message": "Missing 'action' in JSON body"}), 400
        
        action = data['action']
        
        if env._done:
            return jsonify({
                "status": "error", 
                "message": "Episode is already finished. Please hit /api/reset first."
            }), 400
        
        result = asyncio.run(env.step(action))
        
        return jsonify({
            "status": "success",
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
            "state": env.state()  # Send the full state to update the UI visuals comprehensively
        })

    except ValueError as ve:
        # Handling invalid actions gracefully
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/autopilot', methods=['POST'])
def autopilot():
    try:
        if env._done:
            return jsonify({"status": "error", "message": "Episode is already finished."}), 400
            
        obs_dict = env._build_observation()
        obs_obj = AmbuBrainObservation(**obs_dict)
        
        # Use existing inference machinery (hardcode task string for demo)
        action_str, error = asyncio.run(inference.get_llm_action(obs_obj, "emergency_handling"))
        
        if error:
            return jsonify({"status": "error", "message": error}), 500
            
        return jsonify({"status": "success", "action": action_str})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Running on HF recommended port 7860
    app.run(host='0.0.0.0', port=7860, debug=True)

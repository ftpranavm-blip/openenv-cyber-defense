"""
AmbuBrain Inference Script
Runs the LLM agent against the AmbuBrain environment for all three tasks.
Follows strict OpenEnv logging format and Async Execution required by Hackathon baseline.
"""
import os
import sys
import asyncio
import time
import re

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import AmbuBrainEnv, ACTIONS, AmbuBrainAction
from env.task import get_task

API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.5-flash"
HF_TOKEN = "AIzaSyCMDeQdnxaod7la9i3SE--Upxepm5O25sI"
API_KEY = HF_TOKEN

# Client is dynamically created inside the async function to prevent event loop thread deadlock

TASK_NAMES = ["emergency_handling", "routing_optimization", "fairness_ethics"]
MAX_TOTAL_REWARD = 20.0
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    done_str = "true" if done else "false"
    err_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


def build_system_prompt() -> str:
    return (
        "You are an intelligent ambulance routing agent. "
        "Given the current patient and hospital state, you must select the best action.\n"
        "CAREFULLY WEIGH the patient's severity against the hospital's Rating (Stars) and Specialty. "
        "High quality/top-rated hospitals are vital for critical patients. Factor in simulated traffic.\n\n"
        f"Available actions: {ACTIONS}\n\n"
        "Respond with ONLY the exact action string. For example, 'route_to_hospital_A'. Do NOT include code blocks, punctuation, or any other conversational text whatsoever."
    )


def build_user_prompt(obs_obj, task_name: str) -> str:
    # obs_obj is the Pydantic AmbuBrainObservation
    obs = obs_obj.model_dump()
    lines = [
        f"Task: {task_name}",
        f"Patient severity: {obs['patient_severity']}",
        f"Medical Emergency: {obs['patient_condition']}",
        f"Patient zone: {obs['patient_zone']}",
        f"Patient condition score: {obs['patient_condition_score']}",
        "",
        "Hospital A (Lilavati Hospital):",
        f"  Rating: ⭐ {obs['hospital_A_rating']}",
        f"  Specialty: {obs['hospital_A_specialty']}",
        f"  Available beds: {obs['hospital_A_available_beds']}",
        f"  ICU available: {obs['hospital_A_icu_available']}",
        f"  Wait time: {obs['hospital_A_wait_time']} min",
        f"  Travel time: {obs['hospital_A_travel_time']} min",
        "",
        "Hospital B (KEM Hospital):",
        f"  Rating: ⭐ {obs['hospital_B_rating']}",
        f"  Specialty: {obs['hospital_B_specialty']}",
        f"  Available beds: {obs['hospital_B_available_beds']}",
        f"  ICU available: {obs['hospital_B_icu_available']}",
        f"  Wait time: {obs['hospital_B_wait_time']} min",
        f"  Travel time: {obs['hospital_B_travel_time']} min",
        "",
        "Hospital C (Breach Candy):",
        f"  Rating: ⭐ {obs['hospital_C_rating']}",
        f"  Specialty: {obs['hospital_C_specialty']}",
        f"  Available beds: {obs['hospital_C_available_beds']}",
        f"  ICU available: {obs['hospital_C_icu_available']}",
        f"  Wait time: {obs['hospital_C_wait_time']} min",
        f"  Travel time: {obs['hospital_C_travel_time']} min",
        "",
        "Select the best action:",
    ]
    return "\n".join(lines)


def parse_action(response_text: str) -> str:
    text = response_text.strip().lower()
    for action in ACTIONS:
        if action.lower() in text:
            return action
    for letter, action in [("a", "route_to_hospital_A"), ("b", "route_to_hospital_B"), ("c", "route_to_hospital_C")]:
        if f"hospital_{letter}" in text or f"hospital {letter}" in text:
            return action
    return "reroute"


async def get_llm_action(obs_obj, task_name: str) -> tuple:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for attempt in range(6):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_prompt(obs_obj, task_name)},
                ],
                max_tokens=16,
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            action = parse_action(raw)
            return action, None
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower():
                match = re.search(r"retry in (\d+\.?\d*)s", err)
                delay = float(match.group(1)) + 1.0 if match else 12.0
                if delay > 6.0:
                    print(f" [DEBUG] Major Rate Limit ({delay}s). Falling back to heuristic.", flush=True)
                    # Return None for error so the backend cleanly executes "reroute"
                    return "reroute", None
                
                print(f" [DEBUG] Rate Limit: Pausing for {delay:.1f}s...", flush=True)
                await asyncio.sleep(delay)
            else:
                return "reroute", None
    return "reroute", "Max retries exceeded due to rate limits."


async def run_task(task_name: str) -> None:
    env = AmbuBrainEnv(seed=42)
    task = get_task(task_name)

    # Initial Async Reset
    result = await env.reset()
    obs = result.observation

    rewards = []
    step = 0
    done = False
    score = 0.0

    log_start(task=task_name, env=AmbuBrainEnv.ENV_NAME, model=MODEL_NAME)

    try:
        while not done:
            action_str, error = await get_llm_action(obs, task_name)
            
            # Pack action in Pydantic logic Model
            action_pkg = AmbuBrainAction(action=action_str)
            
            # Step Action
            result = await env.step(action_pkg)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            step += 1

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        history = env.get_episode_history()
        score = task.grade(history)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=step, score=score, rewards=rewards)
        print()


async def main() -> None:
    for task_name in TASK_NAMES:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())

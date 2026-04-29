import os
import asyncio
import json
import httpx
from typing import List
from openai import OpenAI

# Read strictly from environment variables
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
API_KEY = os.environ.get("HF_TOKEN")

BENCHMARK = "smart-energy-grid-env"
TASK_NAME = "grid_recovery_pipeline"
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.8
MAX_TOTAL_REWARD = 3.0
ENV_URL = "http://localhost:8080"

# --- REQUIRED LOGGING FUNCTIONS ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

def get_model_message(client, step, system_logs, last_output, history) -> str:
    """Prompt the LLM and return the action string."""
    prompt = f"""
    You are an AI managing an IoT energy grid.
    Logs: {system_logs}
    Output: {last_output}
    History: {history}
    
    Available commands: QUERY_LOGS, RESTART, PATCH, REROUTE.
    Reply ONLY with a valid JSON object matching this schema, nothing else:
    {{"command": "string", "target_meter_id": "string", "payload": "string"}}
    """
    
    try:
        # The OpenAI client will automatically route to Hugging Face based on the base_url
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1 # Low temperature for more deterministic JSON outputs
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"command": "ERROR", "target_meter_id": "none", "payload": "none"}'

async def main() -> None:
    # Initialize the OpenAI Client with your Hugging Face credentials
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient() as http_client:
            reset_resp = await http_client.post(f"{ENV_URL}/reset", timeout=30.0)
            obs = reset_resp.json()["observation"]
            
            last_logs = obs["system_logs"]
            last_output = obs["last_command_output"]
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                # Request action from Hugging Face model
                message = get_model_message(client, step, last_logs, last_output, history)
                
                # Parse the JSON string from the model
                try:
                    action_dict = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[DEBUG] Model did not return valid JSON. Raw output: {message}", flush=True)
                    action_dict = {"command": "ERROR"}

                # Execute OpenENV step()
                step_resp = await http_client.post(f"{ENV_URL}/step", json=action_dict, timeout=30.0)
                step_data = step_resp.json()

                obs = step_data["observation"]
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)

                rewards.append(reward)
                steps_taken = step
                last_logs = obs["system_logs"]
                last_output = obs["last_command_output"]
                last_reward = reward

                log_step(step=step, action=json.dumps(action_dict), reward=reward, done=done, error=None)
                history.append(f"Step {step}: {json.dumps(action_dict)!r} -> reward {reward:+.2f}")

                if done:
                    break

            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = min(max(score, 0.0), 1.0) 
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Environment error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
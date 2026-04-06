import os
import asyncio
import json
import httpx
from typing import List
from openai import AsyncOpenAI

# Strict environment variables required by Meta Hackathon
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

BENCHMARK = "smart-energy-grid-env"
TASK_NAME = "grid_recovery_pipeline"
MAX_STEPS = 15
SUCCESS_SCORE_THRESHOLD = 0.8
MAX_TOTAL_REWARD = 3.0  # 3 tasks maxing out at 1.0 reward each
ENV_URL = "http://localhost:8080"

# --- REQUIRED LOGGING FUNCTIONS ---
# Do not modify the formatting or field names of these functions.
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    print(f"[STEP] step={step} action={action!r} reward={reward:+.2f} done={done} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

def get_model_message(client, step, system_logs, last_output, history) -> str:
    """Prompt the LLM to decide the next action based on the environment state."""
    prompt = f"""
    You are an AI agent managing an IoT energy grid.
    Recent Logs: {system_logs}
    Last Command Output: {last_output}
    History: {history}
    
    Determine your next action. Reply ONLY with a valid JSON object matching this schema:
    {{"command": "string", "target_meter_id": "string or null", "payload": "string or null"}}
    Available commands: QUERY_LOGS, RESTART, PATCH, REROUTE.
    """
    # Note: In a production run, we rely on the LLM's reasoning, but for baseline
    # inference reliability, you might use structured outputs or json mode.
    return prompt

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient() as http_client:
            # OpenENV reset() via HTTP to our FastAPI server
            reset_resp = await http_client.post(f"{ENV_URL}/reset", timeout=30.0)
            reset_data = reset_resp.json()
            
            obs = reset_data["observation"]
            last_logs = obs["system_logs"]
            last_output = obs["last_command_output"]
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                # Request action from LLM
                message_prompt = get_model_message(client, step, last_logs, last_output, history)
                
                try:
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": message_prompt}],
                        response_format={"type": "json_object"}
                    )
                    action_json = response.choices[0].message.content
                    action_dict = json.loads(action_json)
                except Exception as exc:
                    print(f"[DEBUG] Model request failed: {exc}", flush=True)
                    action_dict = {"command": "ERROR"}

                # Execute OpenENV step() via HTTP
                step_resp = await http_client.post(f"{ENV_URL}/step", json=action_dict, timeout=30.0)
                step_data = step_resp.json()

                obs = step_data["observation"]
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)
                error = None

                rewards.append(reward)
                steps_taken = step
                last_logs = obs["system_logs"]
                last_output = obs["last_command_output"]
                last_reward = reward

                # The action parameter in log_step expects the string representation of the action
                log_step(step=step, action=json.dumps(action_dict), reward=reward, done=done, error=error)

                history.append(f"Step {step}: {json.dumps(action_dict)!r} -> reward {reward:+.2f}")

                if done:
                    break

            # Calculate normalized score [0, 1] based on standard hackathon constraints
            score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = min(max(score, 0.0), 1.0) 
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Environment interaction error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
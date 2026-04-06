from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models import Action, Observation, Reward
from src.environment import GridEnvironment

app = FastAPI(title="Smart Energy Grid OpenEnv")

# Instantiate our simulation
env = GridEnvironment()

# OpenEnv Wrapper Models
class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: Observation

@app.post("/reset", response_model=ResetResponse)
async def reset():
    """Initializes a clean state. Required by Phase 1 validation."""
    try:
        obs = env.reset()
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    """Executes the agent's action and advances the environment."""
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward.score,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def state():
    """Returns the raw internal state without advancing the environment."""
    return env.state()
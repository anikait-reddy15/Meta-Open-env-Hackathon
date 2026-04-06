from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class GridState(BaseModel):
    """Internal state tracking (not directly exposed to the agent unless via state())."""
    meters_online: int
    total_meters: int
    system_alerts: List[str]
    current_task: str

class Observation(BaseModel):
    """What the LLM agent sees at each step."""
    system_logs: str = Field(description="Recent logs from the IoT energy meters.")
    active_alarms: List[str] = Field(description="List of currently active critical alarms.")
    last_command_output: str = Field(description="The console output of the last action taken.")

class Action(BaseModel):
    """The commands the LLM agent can execute."""
    command: str = Field(description="The command to run (e.g., 'RESTART', 'QUERY_LOGS', 'PATCH').")
    target_meter_id: Optional[str] = Field(default=None, description="The ID of the specific meter to target.")
    payload: Optional[str] = Field(default=None, description="Optional configuration data or firmware payload.")

class Reward(BaseModel):
    """The grading signal sent back to the agent."""
    score: float = Field(ge=0.0, le=1.0, description="Task score between 0.0 and 1.0.")
    message: str = Field(description="Feedback on partial progress or task completion.")
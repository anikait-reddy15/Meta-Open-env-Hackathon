from src.tasks.base_tasks import BaseTask
from src.models import Action
from typing import Tuple

class Task3Hard(BaseTask):
    def evaluate(self, action: Action, state_data: dict) -> Tuple[float, str, str, str]:
        if action.command == "REROUTE" and action.target_meter_id == "ALL":
            state_data["grid_stabilized"] = True
            return (
                1.0, 
                "Task 3 Complete: Grid stabilized.", 
                "[INFO] Grid power rebalanced. All systems normal.", 
                "SUCCESS: REROUTE executed."
            )
        else:
            return (
                0.0, 
                "Action registered.", 
                "[CRITICAL] Sudden power surge detected on Main Line! Requires immediate REROUTE targeting ALL.", 
                "Command failed or unrecognized."
            )
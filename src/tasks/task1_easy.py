from src.tasks.base_tasks import BaseTask
from src.models import Action
from typing import Tuple

class Task1Easy(BaseTask):
    def evaluate(self, action: Action, state_data: dict) -> Tuple[float, str, str, str]:
        if action.command == "RESTART" and action.target_meter_id == "METER-042":
            state_data["meters_online"] = 100
            return (
                1.0, 
                "Task 1 Complete: Meter restarted.", 
                "[INFO] METER-042 online. [WARN] Subnet Beta running deprecated firmware v1.0.", 
                "SUCCESS: RESTART executed."
            )
        elif action.command == "QUERY_LOGS":
            return (
                0.2, 
                "Logs queried successfully.", 
                "[WARN] Connection timeout from METER-042. Needs RESTART.", 
                "SUCCESS: Logs retrieved."
            )
        else:
            return (
                0.0, 
                "Action registered.", 
                "[WARN] Connection timeout from METER-042 at Substation Alpha.", 
                "Command failed or unrecognized."
            )
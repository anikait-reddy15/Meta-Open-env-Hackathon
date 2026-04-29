from src.tasks.base_tasks import BaseTask
from src.models import Action
from typing import Tuple

class Task2Medium(BaseTask):
    def evaluate(self, action: Action, state_data: dict) -> Tuple[float, str, str, str]:
        if action.command == "PATCH" and action.payload == "v2.0":
            state_data["subnet_beta_patched"] = True
            return (
                1.0, 
                "Task 2 Complete: Firmware patched.", 
                "[INFO] Subnet Beta updated. [CRITICAL] Sudden power surge detected on Main Line!", 
                "SUCCESS: PATCH applied."
            )
        else:
            return (
                0.0, 
                "Action registered.", 
                "[WARN] Subnet Beta running deprecated firmware v1.0. Requires PATCH with payload v2.0.", 
                "Command failed or unrecognized."
            )
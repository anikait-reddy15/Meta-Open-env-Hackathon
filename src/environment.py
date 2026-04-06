from src.models import Observation, Action, Reward, GridState
from typing import Tuple

class GridEnvironment:
    """
    Simulates a smart energy meter grid.
    Tasks:
    1 (Easy): Restart an offline meter based on logs.
    2 (Medium): Apply a firmware patch to a subnet.
    3 (Hard): Reroute power during a grid surge.
    """
    def __init__(self):
        self.max_steps = 15
        self._initialize_state()

    def _initialize_state(self):
        self.current_step = 0
        self.task_level = 1
        self.done = False
        self.state_data = {
            "meters_online": 99,
            "total_meters": 100,
            "offline_meter_id": "METER-042",
            "subnet_beta_patched": False,
            "grid_stabilized": False
        }

    def reset(self) -> Observation:
        """Resets the environment to a clean, initial state."""
        self._initialize_state()
        return Observation(
            system_logs="[WARN] Connection timeout from METER-042 at Substation Alpha. Status: OFFLINE.",
            active_alarms=["METER_OFFLINE"],
            last_command_output="System initialized. Awaiting operator input."
        )

    def state(self) -> GridState:
        """Returns the internal state variables without advancing the simulation."""
        alarms = []
        if self.task_level == 1: alarms.append("METER_OFFLINE")
        if self.task_level == 2: alarms.append("FIRMWARE_MISMATCH")
        if self.task_level == 3: alarms.append("POWER_SURGE")

        return GridState(
            meters_online=self.state_data["meters_online"],
            total_meters=self.state_data["total_meters"],
            system_alerts=alarms,
            current_task=f"Task {self.task_level}"
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """Applies the agent's action, calculates deterministic reward, and transitions state."""
        self.current_step += 1
        reward_score = 0.0
        msg = "Action registered."
        obs_logs = ""
        obs_output = "Command failed or unrecognized."

        # TASK 1: EASY - Restart the offline meter
        if self.task_level == 1:
            if action.command == "RESTART" and action.target_meter_id == "METER-042":
                reward_score = 1.0
                self.task_level = 2
                self.state_data["meters_online"] = 100
                msg = "Task 1 Complete: Meter restarted."
                obs_logs = "[INFO] METER-042 online. [WARN] Subnet Beta running deprecated firmware v1.0."
                obs_output = "SUCCESS: RESTART executed."
            elif action.command == "QUERY_LOGS":
                # Partial progress reward
                reward_score = 0.2
                msg = "Logs queried successfully."
                obs_logs = "[WARN] Connection timeout from METER-042. Needs RESTART."
                obs_output = "SUCCESS: Logs retrieved."
            else:
                obs_logs = "[WARN] Connection timeout from METER-042 at Substation Alpha."

        # TASK 2: MEDIUM - Patch Firmware
        elif self.task_level == 2:
            if action.command == "PATCH" and action.payload == "v2.0":
                reward_score = 1.0
                self.task_level = 3
                self.state_data["subnet_beta_patched"] = True
                msg = "Task 2 Complete: Firmware patched."
                obs_logs = "[INFO] Subnet Beta updated. [CRITICAL] Sudden power surge detected on Main Line!"
                obs_output = "SUCCESS: PATCH applied."
            else:
                obs_logs = "[WARN] Subnet Beta running deprecated firmware v1.0. Requires PATCH with payload v2.0."

        # TASK 3: HARD - Reroute Grid
        elif self.task_level == 3:
            if action.command == "REROUTE" and action.target_meter_id == "ALL":
                reward_score = 1.0
                self.done = True
                self.state_data["grid_stabilized"] = True
                msg = "Task 3 Complete: Grid stabilized."
                obs_logs = "[INFO] Grid power rebalanced. All systems normal."
                obs_output = "SUCCESS: REROUTE executed."
            else:
                obs_logs = "[CRITICAL] Sudden power surge detected on Main Line! Requires immediate REROUTE targeting ALL."

        # Check termination condition
        if self.current_step >= self.max_steps:
            self.done = True
            msg = "Max steps reached. Simulation terminated."

        obs = Observation(
            system_logs=obs_logs,
            active_alarms=["ACTIVE_ALERT"] if not self.done else [],
            last_command_output=obs_output
        )
        
        return obs, Reward(score=reward_score, message=msg), self.done, {"task_level": self.task_level}
from src.models import Observation, Action, Reward, GridState
from src.tasks.task1_easy import Task1Easy
from src.tasks.task2_medium import Task2Medium
from src.tasks.task3_hard import Task3Hard
from typing import Tuple

class GridEnvironment:
    def __init__(self):
        self.max_steps = 15
        self.task_graders = {
            1: Task1Easy(),
            2: Task2Medium(),
            3: Task3Hard()
        }
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
        self._initialize_state()
        return Observation(
            system_logs="[WARN] Connection timeout from METER-042 at Substation Alpha. Status: OFFLINE.",
            active_alarms=["METER_OFFLINE"],
            last_command_output="System initialized. Awaiting operator input."
        )

    def state(self) -> GridState:
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
        self.current_step += 1
        
        # Route the action to the correct isolated task grader
        grader = self.task_graders[self.task_level]
        reward_score, msg, obs_logs, obs_output = grader.evaluate(action, self.state_data)

        # Advance task level if completed
        if reward_score == 1.0:
            if self.task_level < 3:
                self.task_level += 1
            else:
                self.done = True

        if self.current_step >= self.max_steps:
            self.done = True
            msg = "Max steps reached. Simulation terminated."

        obs = Observation(
            system_logs=obs_logs,
            active_alarms=["ACTIVE_ALERT"] if not self.done else [],
            last_command_output=obs_output
        )
        
        return obs, Reward(score=reward_score, message=msg), self.done, {"task_level": self.task_level}
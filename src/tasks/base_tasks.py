from abc import ABC, abstractmethod
from typing import Tuple
from src.models import Action

class BaseTask(ABC):
    """Abstract base class for all OpenEnv grading tasks."""
    
    @abstractmethod
    def evaluate(self, action: Action, state_data: dict) -> Tuple[float, str, str, str]:
        """
        Evaluates the action against the task objectives.
        Returns:
            Tuple containing: (reward_score, feedback_message, new_system_logs, command_output)
        """
        pass
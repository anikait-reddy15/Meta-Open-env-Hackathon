from src.tasks.task1_easy import Task1Easy
from src.tasks.task2_medium import Task2Medium
from src.tasks.task3_hard import Task3Hard
from src.models import Action

def test_task_1_deterministic_grading():
    task = Task1Easy()
    state = {"meters_online": 99}
    
    # Test partial reward logic
    action_query = Action(command="QUERY_LOGS", target_meter_id=None, payload=None)
    reward, msg, logs, output = task.evaluate(action_query, state)
    assert reward == 0.2
    assert "Logs queried" in msg

    # Test full reward logic
    action_restart = Action(command="RESTART", target_meter_id="METER-042", payload=None)
    reward, msg, logs, output = task.evaluate(action_restart, state)
    assert reward == 1.0
    assert state["meters_online"] == 100

def test_task_2_deterministic_grading():
    task = Task2Medium()
    state = {"subnet_beta_patched": False}
    
    action = Action(command="PATCH", target_meter_id=None, payload="v2.0")
    reward, msg, logs, output = task.evaluate(action, state)
    assert reward == 1.0
    assert state["subnet_beta_patched"] is True

def test_task_3_deterministic_grading():
    task = Task3Hard()
    state = {"grid_stabilized": False}
    
    action = Action(command="REROUTE", target_meter_id="ALL", payload=None)
    reward, msg, logs, output = task.evaluate(action, state)
    assert reward == 1.0
    assert state["grid_stabilized"] is True
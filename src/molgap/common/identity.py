"""Strict executable identity checks; no source rewriting or admission policy."""
import re


def validate_execution_identity(spec, arm_id, trajectory, source_commit):
    """Reject unreplayable plans before publication; never rewrite an action."""
    declaration = spec.to_dict()
    state = trajectory.get("state_at_start", {})
    run_id = declaration["logical_run_id"] + ":" + arm_id + ":downstream"
    actions = [action for action in trajectory.get("actions", [])
               if run_id in action.get("run_ids", [])]
    if state.get("source_commit") != source_commit or len(actions) != 1:
        raise ValueError("Prospective executable source/run identity mismatch: " + arm_id)
    action = actions[0]
    if action.get("source_commit") != source_commit:
        raise ValueError("Prospective action/source commit mismatch: " + arm_id)
    prefix = declaration["logical_run_id"] if declaration["prospective"].get("same_run_replay") else arm_id
    attempts = action.get("attempt_ids", [])
    if len(attempts) != 1 or not re.fullmatch(re.escape(prefix) + r"-v[1-9][0-9]*", attempts[0]):
        raise ValueError("Prospective executable attempt identity mismatch: " + arm_id)
    return attempts[0]

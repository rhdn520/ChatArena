#!/usr/bin/env python3
"""Slurm specifications only. No tests, model loads or execution gates."""
import json
import sys
from slurm_standard import ROOT, validate_file

def handle_event(event):
    raw = json.dumps(event.get("tool_input", {}), ensure_ascii=False)
    if "submit_qwen35_games.py" not in raw and "scripts/slurm/run_qwen35_games.sh" not in raw:
        return {}
    try:
        validate_file(ROOT / "scripts/slurm/run_qwen35_games.sh")
    except (ValueError, OSError, KeyError) as exc:
        if event.get("hook_event_name") == "PostToolUse":
            return {"decision": "block", "reason": "Slurm specification: " + str(exc)}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Slurm specification: " + str(exc)}}
    return {}

if __name__ == "__main__":
    print(json.dumps(handle_event(json.load(sys.stdin))))

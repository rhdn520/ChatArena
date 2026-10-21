#!/usr/bin/env python3
"""Slurm specifications only. No tests, model loads or execution gates."""
import json
import sys
from slurm_standard import ROOT, validate_file

def handle_event(event):
    raw = json.dumps(event.get("tool_input", {}), ensure_ascii=False)
    paths = [path for path in (
        "scripts/slurm/run_qwen35_games.sh", "scripts/slurm/run_mixed_games.sh",
        "scripts/slurm/setup_mixed_env.sh",
    ) if path in raw]
    if "submit_qwen35_games.py" in raw and "scripts/slurm/run_qwen35_games.sh" not in paths:
        paths.append("scripts/slurm/run_qwen35_games.sh")
    if not paths:
        return {}
    try:
        for path in paths:
            validate_file(ROOT / path)
    except (ValueError, OSError, KeyError) as exc:
        if event.get("hook_event_name") == "PostToolUse":
            return {"decision": "block", "reason": "Slurm specification: " + str(exc)}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "Slurm specification: " + str(exc)}}
    return {}

if __name__ == "__main__":
    print(json.dumps(handle_event(json.load(sys.stdin))))

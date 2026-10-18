#!/usr/bin/env python3
"""Report the existing workload settings, then submit the live workspace."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / ".codex/hooks"))
from slurm_standard import validate_file
validate_file(root / "scripts/slurm/run_qwen35_games.sh")
policy = json.loads((root / "configs/slurm_standard.json").read_text())
profile = policy["submission_profiles"]["scripts/slurm/run_qwen35_games.sh"]
liar_only = os.environ.get("CHAT_ARENA_LIAR_ONLY") == "1"
print(json.dumps({
    "resources": policy["required_directives"], "partition": "scheduler default (not explicitly set)",
    "python": str(root / profile["python"]), "model": profile["model"],
    "enable_thinking": profile["enable_thinking"], "max_tokens": profile["max_tokens"],
    "thinking_token_budget": profile["thinking_token_budget"],
    "control_thinking_token_budget": profile["control_thinking_token_budget"],
    "sampling_parameters": profile["sampling_parameters"],
    "reasoning_end_str": profile["reasoning_end_str"],
    "request_timeout": profile["request_timeout"],
    "moderator": "not used (Liar only)" if liar_only else "enabled for Mafia; same model",
    "workload": "Liar Game only: 5 players, 1 clue round, seeds 1-10" if liar_only else
                "Liar Game first: 5 players, 1 clue round, seeds 1-10; then Mafia: 7 players, seeds 1-3",
    "logs": str(root / "logs/qwen35_%j.out"),
    "outputs": str(root / "outputs/qwen35_pilot/job_JOB_ID"),
}, ensure_ascii=False, indent=2), flush=True)
subprocess.run(["sbatch", "scripts/slurm/run_qwen35_games.sh"], cwd=root, check=True)

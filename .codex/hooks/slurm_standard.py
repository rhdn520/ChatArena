#!/usr/bin/env python3
"""Slurm directive/resource specifications only, explicitly requested by user."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]

def validate_file(path):
    policy = json.loads((ROOT / "configs/slurm_standard.json").read_text())
    path = Path(path).resolve()
    try:
        relative_path = str(path.relative_to(ROOT))
    except ValueError:
        relative_path = ""
    directives = {}
    for line in Path(path).read_text().splitlines():
        if not line.lstrip().startswith("#SBATCH"):
            continue
        match = re.fullmatch(r"#SBATCH --([\w-]+)=(\S+)", line)
        if not match:
            raise ValueError("Invalid Slurm directive: " + line)
        name, value = match.groups()
        if name not in policy["allowed_directives"] or name in directives:
            raise ValueError("Unknown or duplicate Slurm directive: " + name)
        directives[name] = value
    required = policy.get("resource_profiles", {}).get(relative_path, policy["required_directives"])
    for name, value in required.items():
        if directives.get(name) != value:
            raise ValueError(f"Required Slurm --{name}={value}")
    return directives

if __name__ == "__main__":
    print(json.dumps(validate_file(sys.argv[1])))

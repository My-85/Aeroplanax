#!/usr/bin/env python3
"""
config_patcher.py — Patches REWARD_CONFIG in full_domain_reward.py from a JSON config file.

This is a FIXED utility (do not modify). It reads reward_config.json and writes
the corresponding REWARD_CONFIG dict into the reward Python file.

Usage:
    python config_patcher.py [--config reward_config.json] [--reward-file PATH] [--backup] [--restore]
"""

import json
import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Paths
AUTOTUNER_DIR = Path(__file__).resolve().parent
PLANAX_DIR = AUTOTUNER_DIR.parent / "Planax"
DEFAULT_CONFIG = AUTOTUNER_DIR / "reward_config.json"
DEFAULT_REWARD_FILE = PLANAX_DIR / "envs" / "reward_functions" / "quat_baseline_reward.py"
BACKUP_DIR = AUTOTUNER_DIR / ".backups"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """Load reward config from JSON, stripping comments."""
    with open(config_path, "r") as f:
        data = json.load(f)
    # Remove internal comments
    return {k: v for k, v in data.items() if not k.startswith("_")}


def config_to_python(config: dict) -> str:
    """Convert config dict to Python source code for REWARD_CONFIG."""
    lines = ["REWARD_CONFIG = {"]
    for key, value in config.items():
        if isinstance(value, float):
            lines.append(f'    "{key}": {value},')
        elif isinstance(value, int):
            lines.append(f'    "{key}": {value},')
        elif isinstance(value, str):
            lines.append(f'    "{key}": "{value}",')
        else:
            lines.append(f'    "{key}": {value},')
    lines.append("}")
    return "\n".join(lines)


def patch_reward_file(config: dict, reward_file: Path = DEFAULT_REWARD_FILE) -> bool:
    """Replace REWARD_CONFIG dict in the reward file with values from config.

    Returns True if successful, False otherwise.
    """
    content = reward_file.read_text(encoding="utf-8")

    # Find the REWARD_CONFIG = { ... } block
    # Match from 'REWARD_CONFIG = {' to the closing '}' that's at column 0
    pattern = re.compile(
        r"(REWARD_CONFIG\s*=\s*\{)"  # opening
        r"(.*?)"                      # content (non-greedy)
        r"(\n\})",                    # closing brace at start of line
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        print("ERROR: Could not find REWARD_CONFIG dict in", reward_file)
        return False

    new_config_block = config_to_python(config)
    new_content = content[: match.start()] + new_config_block + content[match.end():]

    reward_file.write_text(new_content, encoding="utf-8")
    print(f"OK: Patched REWARD_CONFIG in {reward_file} ({len(config)} keys)")
    return True


def backup_reward_file(reward_file: Path = DEFAULT_REWARD_FILE) -> Path:
    """Create a timestamped backup of the reward file."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"quat_baseline_reward_{ts}.py"
    shutil.copy2(reward_file, backup_path)
    print(f"OK: Backup saved to {backup_path}")
    return backup_path


def restore_reward_file(reward_file: Path = DEFAULT_REWARD_FILE) -> bool:
    """Restore the most recent backup."""
    if not BACKUP_DIR.exists():
        print("ERROR: No backups found")
        return False
    backups = sorted(BACKUP_DIR.glob("quat_baseline_reward_*.py"))
    if not backups:
        print("ERROR: No backups found")
        return False
    latest = backups[-1]
    shutil.copy2(latest, reward_file)
    print(f"OK: Restored from {latest}")
    return True


def validate_config(config: dict) -> list:
    """Check config for obvious issues. Returns list of warnings."""
    warnings = []
    # Check required keys exist for quat baseline reward
    required = [
        "theta_scale_deg", "speed_error_scale", "w_att", "w_speed",
    ]
    for key in required:
        if key not in config:
            warnings.append(f"MISSING required key: {key}")

    # Check weights sum to 1
    w_att = config.get("w_att", 0)
    w_spd = config.get("w_speed", 0)
    if abs(w_att + w_spd - 1.0) > 0.01:
        warnings.append(f"w_att({w_att}) + w_speed({w_spd}) = {w_att+w_spd}, should be ~1.0")

    # Check theta_scale_deg is positive
    ts = config.get("theta_scale_deg", 0)
    if ts <= 0:
        warnings.append(f"theta_scale_deg={ts} should be positive")

    # Check speed_error_scale is positive
    ss = config.get("speed_error_scale", 0)
    if ss <= 0:
        warnings.append(f"speed_error_scale={ss} should be positive")

    return warnings


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch REWARD_CONFIG from JSON")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reward-file", type=Path, default=DEFAULT_REWARD_FILE)
    parser.add_argument("--backup", action="store_true", help="Backup before patching")
    parser.add_argument("--restore", action="store_true", help="Restore most recent backup")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't patch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written")
    args = parser.parse_args()

    if args.restore:
        restore_reward_file(args.reward_file)
        sys.exit(0)

    config = load_config(args.config)
    warnings = validate_config(config)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
        if any("MISSING" in w for w in warnings):
            print("FATAL: Missing required keys, aborting")
            sys.exit(1)

    if args.validate_only:
        print("Config valid" if not warnings else "Config has warnings (see above)")
        sys.exit(0)

    if args.dry_run:
        print(config_to_python(config))
        sys.exit(0)

    if args.backup:
        backup_reward_file(args.reward_file)

    success = patch_reward_file(config, args.reward_file)
    sys.exit(0 if success else 1)

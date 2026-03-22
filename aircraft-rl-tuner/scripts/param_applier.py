#!/usr/bin/env python3
"""
param_applier.py
================
Deterministic regex-based parameter replacement.

Instead of having Claude rewrite entire files (error-prone for large JAX code),
this module only modifies numerical values in well-known locations:

  - reward_config:     "key": OLD  →  "key": NEW     in REWARD_CONFIG dict
  - training_config:   "KEY": OLD  →  "KEY": NEW     in config dict
  - func_default:      arg=OLD     →  arg=NEW        in function signature
  - dataclass_default: field: type = OLD → field: type = NEW  in dataclass

Supports snapshot/rollback so every change is reversible.
"""

import re
import json
import copy
from pathlib import Path
from typing import Any
from datetime import datetime

from param_registry import PARAM_REGISTRY

_SCRIPT_DIR = Path(__file__).resolve().parent
_SKILL_ROOT = _SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Per-type replacement functions
# ---------------------------------------------------------------------------

def _format_value(value: Any) -> str:
    """Format a Python value for source-code insertion."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Use scientific notation for very large/small numbers
        if abs(value) >= 1e6 or (0 < abs(value) < 1e-4):
            return f"{value:.0e}" if value == int(value) else f"{value:g}"
        # Ensure at least one decimal for floats
        s = f"{value:g}"
        if "." not in s and "e" not in s.lower():
            s += ".0"
        return s
    return repr(value)


def _apply_reward_config(text: str, key: str, value: Any) -> tuple[str, bool]:
    """Replace value in REWARD_CONFIG dict: "key": OLD → "key": NEW."""
    # The key in REWARD_CONFIG is the part after "reward."
    cfg_key = key.split(".", 1)[1] if "." in key else key
    # Match: "cfg_key": <number>
    pattern = rf'("{cfg_key}"\s*:\s*)([^\s,}}]+)'
    new_val = _format_value(value)

    new_text, count = re.subn(pattern, rf'\g<1>{new_val}', text, count=1)
    return new_text, count > 0


def _apply_training_config(text: str, key: str, value: Any) -> tuple[str, bool]:
    """Replace value in training config dict: "KEY": OLD → "KEY": NEW."""
    cfg_key = key.split(".", 1)[1] if "." in key else key
    # Match: "KEY": <value>  (handles int, float, bool, scientific notation)
    pattern = rf'("{cfg_key}"\s*:\s*)([^\s,}}]+)'
    new_val = _format_value(value)

    new_text, count = re.subn(pattern, rf'\g<1>{new_val}', text, count=1)
    return new_text, count > 0


def _apply_func_default(text: str, key: str, value: Any, info: dict) -> tuple[str, bool]:
    """Replace function parameter default: arg=OLD → arg=NEW."""
    arg_name = info["arg_name"]
    # Match: arg_name: type = OLD  or  arg_name=OLD
    pattern = rf'({arg_name}\s*(?::\s*\w+\s*)?=\s*)([^\s,\)]+)'
    new_val = _format_value(value)

    new_text, count = re.subn(pattern, rf'\g<1>{new_val}', text, count=1)
    return new_text, count > 0


def _apply_dataclass_default(text: str, key: str, value: Any, info: dict) -> tuple[str, bool]:
    """Replace dataclass field default: field: type = OLD → field: type = NEW."""
    field_name = info["field_name"]
    # Match: field_name: SomeType = OLD
    pattern = rf'({field_name}\s*:\s*\w+\s*=\s*)([^\s\n#]+)'
    new_val = _format_value(value)

    new_text, count = re.subn(pattern, rf'\g<1>{new_val}', text, count=1)
    return new_text, count > 0


# ---------------------------------------------------------------------------
# Read current value from file
# ---------------------------------------------------------------------------

def _read_reward_config_value(text: str, key: str):
    """Read current value of a reward config key."""
    cfg_key = key.split(".", 1)[1] if "." in key else key
    pattern = rf'"{cfg_key}"\s*:\s*([^\s,}}]+)'
    m = re.search(pattern, text)
    if m:
        return _parse_value(m.group(1))
    return None


def _read_training_config_value(text: str, key: str):
    """Read current value of a training config key."""
    cfg_key = key.split(".", 1)[1] if "." in key else key
    pattern = rf'"{cfg_key}"\s*:\s*([^\s,}}]+)'
    m = re.search(pattern, text)
    if m:
        return _parse_value(m.group(1))
    return None


def _read_func_default_value(text: str, info: dict):
    """Read current value of a function default parameter."""
    arg_name = info["arg_name"]
    pattern = rf'{arg_name}\s*(?::\s*\w+\s*)?=\s*([^\s,\)]+)'
    m = re.search(pattern, text)
    if m:
        return _parse_value(m.group(1))
    return None


def _read_dataclass_default_value(text: str, info: dict):
    """Read current value of a dataclass field default."""
    field_name = info["field_name"]
    pattern = rf'{field_name}\s*:\s*\w+\s*=\s*([^\s\n#]+)'
    m = re.search(pattern, text)
    if m:
        return _parse_value(m.group(1))
    return None


def _parse_value(s: str) -> Any:
    """Parse a string value from source code into Python type."""
    s = s.strip().rstrip(",")
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_params(changes: dict) -> tuple[list[str], list[str]]:
    """
    Apply parameter changes to source files.

    Args:
        changes: dict of {param_name: new_value}, e.g. {"reward.alive_bonus": 0.05}

    Returns:
        (applied, errors) — lists of param names that were applied or failed.
    """
    applied = []
    errors = []

    # Group changes by file to minimize I/O
    by_file = {}
    for key, value in changes.items():
        if key not in PARAM_REGISTRY:
            errors.append(f"Unknown parameter: {key}")
            continue
        info = PARAM_REGISTRY[key]
        fpath = info["file"]
        if fpath not in by_file:
            by_file[fpath] = []
        by_file[fpath].append((key, value, info))

    for fpath, params in by_file.items():
        path = Path(fpath)
        if not path.exists():
            for key, _, _ in params:
                errors.append(f"{key}: file not found: {fpath}")
            continue

        text = path.read_text(encoding="utf-8")
        original = text

        for key, value, info in params:
            ptype = info["type"]
            if ptype == "reward_config":
                text, ok = _apply_reward_config(text, key, value)
            elif ptype == "training_config":
                text, ok = _apply_training_config(text, key, value)
            elif ptype == "func_default":
                text, ok = _apply_func_default(text, key, value, info)
            elif ptype == "dataclass_default":
                text, ok = _apply_dataclass_default(text, key, value, info)
            else:
                errors.append(f"{key}: unknown type '{ptype}'")
                continue

            if ok:
                applied.append(key)
                print(f"  [apply] {key} = {value}")
            else:
                errors.append(f"{key}: regex did not match in {path.name}")

        if text != original:
            path.write_text(text, encoding="utf-8")

    return applied, errors


def snapshot_current() -> dict:
    """Read all current parameter values from source files. Used for rollback."""
    snapshot = {}

    # Cache file contents to avoid re-reading
    file_cache = {}

    for key, info in PARAM_REGISTRY.items():
        fpath = info["file"]
        if fpath not in file_cache:
            path = Path(fpath)
            if path.exists():
                file_cache[fpath] = path.read_text(encoding="utf-8")
            else:
                file_cache[fpath] = None

        text = file_cache[fpath]
        if text is None:
            continue

        ptype = info["type"]
        if ptype == "reward_config":
            val = _read_reward_config_value(text, key)
        elif ptype == "training_config":
            val = _read_training_config_value(text, key)
        elif ptype == "func_default":
            val = _read_func_default_value(text, info)
        elif ptype == "dataclass_default":
            val = _read_dataclass_default_value(text, info)
        else:
            continue

        if val is not None:
            snapshot[key] = val

    return snapshot


def rollback(snapshot: dict):
    """Restore parameter values from a snapshot."""
    if not snapshot:
        print("[rollback] Empty snapshot — nothing to restore.")
        return
    print(f"[rollback] Restoring {len(snapshot)} parameter(s)...")
    applied, errors = apply_params(snapshot)
    if errors:
        print(f"[rollback] WARNING: {len(errors)} error(s) during rollback:")
        for e in errors:
            print(f"  - {e}")
    print(f"[rollback] Restored {len(applied)} parameter(s).")


def save_snapshot(snapshot: dict, iteration: int) -> Path:
    """Save snapshot to JSON file for persistence."""
    snap_dir = _SKILL_ROOT / "assets" / "param_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_file = snap_dir / f"snapshot_iter_{iteration}.json"
    with open(snap_file, "w", encoding="utf-8") as f:
        json.dump({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "params": snapshot,
        }, f, indent=2)
    print(f"[snapshot] Saved to {snap_file}")
    return snap_file


def load_snapshot(iteration: int) -> dict:
    """Load snapshot from JSON file."""
    snap_file = _SKILL_ROOT / "assets" / "param_snapshots" / f"snapshot_iter_{iteration}.json"
    if not snap_file.exists():
        print(f"[snapshot] Not found: {snap_file}")
        return {}
    with open(snap_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("params", {})

#!/usr/bin/env python3
"""Minimal test to verify experiment_runner framework changes."""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_summarize_training_log():
    """Test the _summarize_training_log function."""
    from experiment_runner import _summarize_training_log

    # Mock training log
    log_lines = [
        "env_step=1000000 episodic_return=150.5 loss=0.0045 entropy=0.85",
        "curriculum level=2",
        "env_step=2000000 episodic_return=180.3 loss=0.0038 entropy=0.78",
        "curriculum level=3",
        "env_step=3000000 episodic_return=220.1 loss=0.0025 entropy=0.72",
        "curriculum level=4",
    ]

    return_history = [150.5, 160.2, 170.8, 180.3, 190.5, 200.2, 210.5, 220.1]

    summary = _summarize_training_log(log_lines, return_history)

    print("✓ Training summary generated:")
    print(json.dumps(summary, indent=2))

    # Verify key fields exist
    assert "return" in summary, "Missing 'return' field"
    assert "curriculum" in summary, "Missing 'curriculum' field"
    assert "loss" in summary, "Missing 'loss' field"
    assert summary["curriculum"]["max_level_reached"] == 4, "Curriculum level extraction failed"

    print("✓ All assertions passed\n")
    return summary


def test_prompt_building():
    """Test that prompt building works with new training summary."""
    from experiment_runner import _build_claude_prompt

    # Mock champion
    champion = {
        "experiment_id": 0,
        "config_snapshot": {"theta_scale_deg": 30.0, "w_att": 0.7},
        "metrics": {
            "mean_ss_theta": 73.09,
            "mean_ss_dvt": 20.19,
            "crash_rate": 0.25,
            "settled_rate": 0.45,
            "mean_theta_std": 10.80,
            "mean_action_change_rate": 3.5875
        }
    }

    # Mock history with training summary
    history = [
        {
            "experiment_id": 1,
            "status": "discard",
            "description": "Test experiment",
            "config_snapshot": {"theta_scale_deg": 35.0, "w_att": 0.8},
            "metrics": {
                "eval": {
                    "mean_ss_theta": 75.0,
                    "mean_ss_dvt": 21.0,
                    "crash_rate": 0.28,
                    "settled_rate": 0.42,
                    "mean_theta_std": 12.0,
                    "mean_action_change_rate": 4.0
                },
                "training_summary": {
                    "total_records": 50,
                    "return": {
                        "final": 220.5,
                        "converged": True
                    },
                    "curriculum": {
                        "max_level_reached": 4,
                        "final_level": 4
                    },
                    "loss": {
                        "final": 0.0025,
                        "trend": "decreasing"
                    },
                    "anomalies": []
                }
            }
        }
    ]

    config = {"theta_scale_deg": 30.0, "w_att": 0.7}

    prompt = _build_claude_prompt(config, champion, history, iteration=1)

    print("✓ Prompt generated successfully")
    print(f"✓ Prompt length: {len(prompt)} chars")

    # Verify training info is in prompt
    assert "train:" in prompt, "Training summary not in prompt"
    assert "ret=" in prompt, "Return not in prompt"
    assert "conv=" in prompt, "Convergence status not in prompt"
    assert "L4" in prompt, "Curriculum level not in prompt"

    print("✓ Training summary correctly embedded in prompt\n")
    return prompt


if __name__ == "__main__":
    print("=" * 60)
    print("Testing experiment_runner framework changes")
    print("=" * 60)
    print()

    try:
        print("[1/2] Testing _summarize_training_log()...")
        test_summarize_training_log()

        print("[2/2] Testing prompt building with training summary...")
        test_prompt_building()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

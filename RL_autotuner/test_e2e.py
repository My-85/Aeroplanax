#!/usr/bin/env python3
"""End-to-end test for experiment_runner workflow."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

def test_response_parsing():
    """Test Phase 2a response parsing."""
    from experiment_runner import _parse_claude_response

    # Mock Claude response
    mock_response = """
Based on the history, I'll try increasing theta_scale_deg to improve large angle tracking.

```json
{
  "theta_scale_deg": 35.0,
  "speed_error_scale": 40.0,
  "w_att": 0.7,
  "w_speed": 0.3
}
```

This should provide better gradient at large angles.
"""

    config, description = _parse_claude_response(mock_response)

    assert config is not None, "Failed to parse config"
    assert config["theta_scale_deg"] == 35.0, "Wrong theta_scale_deg"
    assert config["w_att"] == 0.7, "Wrong w_att"
    assert description is not None, "Description is None"

    print("✓ Phase 2a response parsing works")
    return config, description


def test_phase2b_response_parsing():
    """Test Phase 2b response parsing."""
    from experiment_runner import _parse_claude_response_phase2b

    mock_response = """
I'll modify the reward function to use curriculum-adaptive scaling.

```python
# REWARD_CONFIG
REWARD_CONFIG = {
    "theta_scale_deg": 30.0,
    "speed_error_scale": 40.0,
    "w_att": 0.7,
    "w_speed": 0.3
}

def quat_baseline_reward_fn(state, params, agent_id, reward_scale):
    curriculum_level = state.curriculum_level
    theta_scales = [25.0, 30.0, 40.0, 50.0, 60.0, 80.0]
    theta_scale = theta_scales[min(curriculum_level, 5)]

    att_r = jnp.exp(-(theta / theta_scale) ** 4)
    speed_r = jnp.exp(-(delta_vt / 40.0) ** 2)

    return att_r ** 0.7 * speed_r ** 0.3
```

```json
{
  "theta_scale_deg": 30.0,
  "speed_error_scale": 40.0,
  "w_att": 0.7,
  "w_speed": 0.3
}
```

This adapts to curriculum levels.
"""

    code, config, description = _parse_claude_response_phase2b(mock_response)

    assert code is not None, "Failed to parse code"
    assert "curriculum_level" in code, "Code missing curriculum_level"
    assert "quat_baseline_reward_fn" in code, "Code missing required function"
    assert config is not None, "Failed to parse config"
    assert config["theta_scale_deg"] == 30.0, "Config parsing failed"

    print("✓ Phase 2b response parsing works")
    return code, config, description


def test_keep_discard_logic():
    """Test is_better_than_champion logic."""
    from experiment_runner import is_better_than_champion

    champion = {
        "metrics": {
            "mean_ss_theta": 73.0,
            "settled_rate": 0.45,
            "crash_rate": 0.25,
            "mean_action_change_rate": 3.5
        }
    }

    # Test 1: Good improvement - should keep
    good_metrics = {
        "mean_ss_theta": 70.0,  # -3° improvement
        "settled_rate": 0.46,    # +0.01 (within tolerance)
        "crash_rate": 0.26,      # +0.01 (within tolerance)
        "mean_action_change_rate": 3.6  # +0.1 (within 20%)
    }
    assert is_better_than_champion(good_metrics, champion), "Should keep good improvement"
    print("✓ Keep logic: good improvement detected")

    # Test 2: Theta improves but settled drops - should discard
    false_progress = {
        "mean_ss_theta": 70.0,  # -3° improvement
        "settled_rate": 0.30,    # -0.15 (>10% drop) - FALSE PROGRESS!
        "crash_rate": 0.25,
        "mean_action_change_rate": 3.5
    }
    assert not is_better_than_champion(false_progress, champion), "Should discard false progress"
    print("✓ Discard logic: false progress detected (settled_rate drop)")

    # Test 3: Theta improves but crash increases - should discard
    high_crash = {
        "mean_ss_theta": 70.0,
        "settled_rate": 0.45,
        "crash_rate": 0.40,      # +0.15 (>10% increase)
        "mean_action_change_rate": 3.5
    }
    assert not is_better_than_champion(high_crash, champion), "Should discard high crash"
    print("✓ Discard logic: high crash rate detected")

    # Test 4: Theta improves but action oscillation increases - should discard
    high_oscillation = {
        "mean_ss_theta": 70.0,
        "settled_rate": 0.45,
        "crash_rate": 0.25,
        "mean_action_change_rate": 4.5  # +1.0 (>20% increase)
    }
    assert not is_better_than_champion(high_oscillation, champion), "Should discard high oscillation"
    print("✓ Discard logic: high action oscillation detected")

    print("✓ All keep/discard logic tests passed")


def test_training_summary_in_results():
    """Test that training summary is correctly saved to results."""
    from experiment_runner import log_result
    import tempfile
    import os

    # Create temp results file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_results = f.name

    try:
        # Patch RESULTS_PATH
        with patch('experiment_runner.RESULTS_PATH', Path(temp_results)):
            config = {"theta_scale_deg": 30.0}
            metrics = {
                "eval": {"mean_ss_theta": 70.0},
                "training_summary": {
                    "total_records": 50,
                    "return": {"final": 220.5, "converged": True},
                    "curriculum": {"max_level_reached": 4},
                    "loss": {"final": 0.0025, "trend": "decreasing"}
                }
            }

            log_result(1, config, metrics, "keep", "Test experiment")

            # Read back and verify
            with open(temp_results) as f:
                line = f.readline()
                result = json.loads(line)

            assert "training_summary" in result["metrics"], "Training summary not saved"
            assert result["metrics"]["training_summary"]["return"]["final"] == 220.5, "Training data corrupted"

            print("✓ Training summary correctly saved to results.jsonl")

    finally:
        os.unlink(temp_results)


if __name__ == "__main__":
    print("=" * 60)
    print("End-to-End Framework Test")
    print("=" * 60)
    print()

    try:
        print("[1/5] Testing Phase 2a response parsing...")
        test_response_parsing()
        print()

        print("[2/5] Testing Phase 2b response parsing...")
        test_phase2b_response_parsing()
        print()

        print("[3/5] Testing keep/discard logic...")
        test_keep_discard_logic()
        print()

        print("[4/5] Testing training summary persistence...")
        test_training_summary_in_results()
        print()

        print("=" * 60)
        print("✓ ALL E2E TESTS PASSED")
        print("=" * 60)
        print()
        print("Framework is ready for production use!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

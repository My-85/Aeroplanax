"""
Large-batch quaternion baseline environment entry point.

This module intentionally keeps the low-level quaternion attitude-tracking
task unchanged and only gives the large-scale rollout script a separate
top-level env name to import.
"""

from envs.aeroplanax_heading_pitch_V_quaternion_version_add_full_roll import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)


class AeroPlanaxQuatLargeBatchEnv(AeroPlanaxHeading_Pitch_V_Env):
    """Dedicated top-level env alias for large batched quaternion rollouts."""


QuatLargeBatchTaskParams = Heading_Pitch_V_TaskParams


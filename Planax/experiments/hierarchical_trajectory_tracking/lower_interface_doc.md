# Lower-Layer Quaternion Baseline Interface

## Checkpoint

```
results/heading_pitch_V_discrete_rnn_2026-05-13-21-17/checkpoints/checkpoint_epoch_600
```

## Network Architecture

- Type: ActorCriticRNN (Flax)
- RNN: GRU, hidden dim = 128, scanned over time
- FC layers: 128 → 256 → 128 → heads
- Activation: ReLU
- LayerNorm on FC2

## Observation Space (21 dims)

| Index | Feature | Description |
|:-----:|---------|-------------|
| 0-2 | qv | Quaternion error vector part (current→target attitude) |
| 3 | dvt | Normalised speed error: (vt - target_vt) / 340 |
| 4 | alt_n | Altitude / 5000 |
| 5 | vt_n | Airspeed / 340 |
| 6-8 | v_b | Target direction in body frame |
| 9-11 | P, Q, R | Angular rates (rad/s) |
| 12-13 | sin/cos(alpha) | Angle of attack |
| 14-15 | sin/cos(beta) | Sideslip angle |
| 16 | prev_throttle | Previous throttle cmd (0-1) |
| 17 | prev_elevator | Previous elevator cmd (-1 to 1) |
| 18 | prev_aileron | Previous aileron cmd (-1 to 1) |
| 19 | prev_rudder | Previous rudder cmd (-1 to 1) |
| 20 | prev_speed_brake | Previous speed brake cmd (0-1) |

## Action Space

Discrete: throttle(31) × elevator(41) × aileron(41) × rudder(41) × speed_brake(5)

Decoding:
```python
norm_thr = thr_idx / 30.0                    # [0, 1]
norm_el  = el_idx * 2.0 / 40.0 - 1.0         # [-1, 1]
norm_ail = ail_idx * 2.0 / 40.0 - 1.0        # [-1, 1]
norm_rud = rud_idx * 2.0 / 40.0 - 1.0        # [-1, 1]
norm_sb  = sb_idx / 4.0                       # [0, 1]
```

## Target Interface

The lower policy is driven by four state fields:

| Field | Type | Range | Description |
|-------|------|:-----:|-------------|
| `state.target_heading` | float32[B] | [-π, π] | Target yaw angle (rad) |
| `state.target_pitch` | float32[B] | [-89°, 89°] | Target pitch angle (rad) |
| `state.target_roll` | float32[B] | [-π, π] | Target roll angle (rad) |
| `state.target_vt` | float32[B] | [120, 360] | Target airspeed (m/s) |

## How Targets Enter Observation

### qv (dims 0-2): Quaternion Error

```python
q_err = _quat_err_bn(q_curr, target_heading, target_pitch, target_roll)
# q_err = q_tgt ⊗ conj(q_curr), normalised with w ≥ 0
qv = q_err[1:4]  # vector part, clipped to [-1, 1]
```

The quaternion error encodes heading, pitch, AND roll error
simultaneously in a single 3-vector.

### v_b (dims 6-8): Body-Frame Target Direction

```python
# NED unit vector pointing in target direction
v_n = [cos(pitch_t)*cos(yaw_t), cos(pitch_t)*sin(yaw_t), -sin(pitch_t)]

# Rotated from NED to body frame
v_b = q_curr ⊗ [0, v_n] ⊗ q_curr*   (quaternion rotation)
v_b = v_b[1:4]  # vector part
```

v_b tells the policy "where is the target direction relative to my nose."

### dvt (dim 3): Speed Error

```python
dvt = (vt - target_vt) / 340.0
```

## Does the Lower Policy Support Direct Target Quaternion?

No. The interface exclusively uses (heading, pitch, roll, vt).
The env converts these to observation-space encoding internally.

However, the encoding IS quaternion-based — setting
target_heading/pitch/roll produces a quaternion error via _quat_err_bn.

## Full Call Chain

```
Upper guidance sets:
  state.target_heading, state.target_pitch,
  state.target_roll, state.target_vt

env._get_obs encodes:
  qv = quaternion_error(q_curr, target_h, target_p, target_r)
  v_b = body_frame_target_direction(q_curr, target_h, target_p)
  dvt = (vt - target_vt) / 340
  + flight state + past action

→ 21-dim observation vector

Frozen ActorCriticRNN:
  → discrete action distribution
  → argmax (or sample) → thr, el, ail, rud, sb indices

env._decode_discrete_actions → normalised commands
env.step → F-16 dynamics update
```

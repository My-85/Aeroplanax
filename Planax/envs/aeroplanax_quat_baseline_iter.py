from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .reward_functions import (
    heading_reward_fn,
    quat_baseline_reward_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
    heading_pitch_v_event_driven_reward_fn,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_quat_baseline_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class Heading_Pitch_V_TaskState(EnvState):
    target_heading: ArrayLike 
    target_pitch: ArrayLike  # 新增目标俯仰角
    target_vt: ArrayLike
    target_roll: ArrayLike   # <--- [新增] 目标滚转角
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,  # 必须包含这一行
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=extra_state[0],
            target_pitch=extra_state[1],  # 新增
            target_roll=extra_state[2], # <--- [修改] 读取 extra_state[2]
            target_vt=extra_state[3],   # <--- [修改] 索引顺延为 3
            last_check_time=env_state.time,
            heading_turn_counts=0,
        )


@struct.dataclass(frozen=True)
class Heading_Pitch_V_TaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0

    #####################################################################################
    max_vt: float = 360.0  # <-- 原值
    # max_vt: float = 600.0    # [修改] 扩大到 600m/s (约 Mach 1.8)，覆盖俯冲极值
    
    min_vt: float = 120.0  # <-- 原值
    # min_vt: float = 80.0     # [修改] 稍微降低下限，让它也见过筋斗顶点的低速状态
    # max_velocities_u_increment: float = 50.0
    # 允许更大的速度指令变化，以便 Agent 能更快地探索到高速区
    max_velocities_u_increment: float = 50.0 # <-- 原值
    # max_velocities_u_increment: float = 100.0
    #####################################################################################


    max_heading_increment: float = jnp.pi/2  # 最大航向变化量(90°)
    max_pitch_increment: float = jnp.pi/6  # 最大俯仰角变化量(30°)
    # [新增] 滚转角变化量限制，建议设大一点，让它学会在大滚转间切换
    max_roll_increment: float = jnp.pi / 2  # 90°
    max_altitude_increment: float = 2100.0

    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000 # 编队最小安全间距

# ---------------- Quaternion helpers (BN: NED -> Body) ----------------
def _quat_normalize(q):
    return q / (jnp.linalg.norm(q) + 1e-9)

def _quat_conj(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_from_euler_bn(roll, pitch, yaw):
    cr, sr = jnp.cos(0.5*roll),  jnp.sin(0.5*roll)
    cp, sp = jnp.cos(0.5*pitch), jnp.sin(0.5*pitch)
    cy, sy = jnp.cos(0.5*yaw),   jnp.sin(0.5*yaw)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return jnp.array([qw, qx, qy, qz])  # NB

def _target_q_nb_from_heading_pitch(yaw_t, pitch_t, roll_t=0.0):
    return _quat_conj(_quat_from_euler_bn(roll_t, pitch_t, yaw_t))  # BN

def _quat_err_nb(q_curr_nb, yaw_t, pitch_t, roll_t):
    q_curr_nb = _quat_normalize(q_curr_nb)
    q_tgt_nb  = _target_q_nb_from_heading_pitch(yaw_t, pitch_t, roll_t)
    q_err = _quat_mul(q_tgt_nb, _quat_conj(q_curr_nb))
    # 消歧：w >= 0
    q_err = jnp.where(q_err[0] < 0.0, -q_err, q_err)
    return q_err  # [w,x,y,z], 最短旋转

def _rotate_ned_to_body(q_nb, v_n):
    """v_b = q_nb * (0,v_n) * q_nb^*"""
    q_nb = _quat_normalize(q_nb)
    p = jnp.array([0.0, v_n[0], v_n[1], v_n[2]])
    qp = _quat_mul(q_nb, p)
    qpq = _quat_mul(qp, _quat_conj(q_nb))
    return qpq[1:]


class AeroPlanaxHeading_Pitch_V_Env(AeroPlanaxEnv[Heading_Pitch_V_TaskState, Heading_Pitch_V_TaskParams]):
    def __init__(self, env_params: Optional[Heading_Pitch_V_TaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(quat_baseline_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            # functools.partial(event_driven_reward_fn, fail_reward=-20, success_reward=20),
            # functools.partial(heading_pitch_v_event_driven_reward_fn, fail_reward=-50, success_reward=50),
        ]

        # 与 reward_functions 一一对应，表示这些奖励是否做势能差分
        # 这里全部设为 False 即可
        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_quat_baseline_fn,
        ]

        # 课程学习：
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)
        # 前5个元素是 [0.2, 0.4, 0.6, 0.8, 1.0]
        # 后10个元素是 [1.0] 重复10次
        # 该数组用于控制航向/俯仰/速度变化量的增量系数
        # 每次 heading_turn_counts 增加时，会按索引取对应的系数值进行缩放
        # 前5次任务切换时增量系数逐步增大（0.2→1.0），后续保持1.0不变

    def _get_obs_size(self) -> int:
        return 16  # 观测维度为22
        # return 28  # 原来是 16，现在扩展到 28

    @property
    def default_params(self) -> Heading_Pitch_V_TaskParams:
        return Heading_Pitch_V_TaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: Heading_Pitch_V_TaskParams,
    ) -> Heading_Pitch_V_TaskState:
        state = super()._init_state(key, params)
        
        # 随机生成初始航向角 (0 到 2π)
        key, key_heading = jax.random.split(key)
        # initial_heading = jnp.full((self.num_agents,), -jnp.pi)
        initial_heading = jax.random.uniform(
            key_heading, 
            shape=(self.num_agents,), 
            minval=0.0, 
            maxval=2.0 * jnp.pi
        )
        
        # 设置初始速度
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        # 修改后：加入随机 Pitch 初始化
        # 随机俯仰角变化量，考虑高度限制
        # 根据当前高度限制俯仰角变化范围
        current_altitude = state.plane_state.altitude
        # 如果高度接近上限，限制俯仰角为负值（向下）
        max_pitch = jnp.where(
            current_altitude > params.max_altitude - 1000,
            -params.max_pitch_increment * 0.5,  # 限制为负值，且幅度减半
            params.max_pitch_increment
        )
        # 如果高度接近下限，限制俯仰角为正值（向上）
        min_pitch = jnp.where(
            current_altitude < params.min_altitude + 1000,
            params.max_pitch_increment * 0.5,  # 限制为正值，且幅度减半
            -params.max_pitch_increment
        )
        key_pitch, key_heading = jax.random.split(key, 2)
        delta_pitch = jax.random.uniform(key_pitch, shape=(self.num_agents,), minval=min_pitch, maxval=max_pitch)
        
        # 计算新的俯仰角，并限制在安全范围内
        new_pitch = state.plane_state.pitch + delta_pitch
        # 限制最终俯仰角在安全范围内（通常在-89到+89度之间）
        safe_pitch_min = jnp.radians(-89.0)  # -89度
        safe_pitch_max = jnp.radians(89.0)   # +89度
        new_pitch = jnp.clip(new_pitch, safe_pitch_min, safe_pitch_max)
        # 随机生成 pitch (-89 ~ +89)
        rand_pitch = jax.random.uniform(key_pitch, shape=(self.num_agents,), minval=safe_pitch_min, maxval=safe_pitch_max)
        
        # 4. [新增] 随机滚转 (Roll) - 全域随机
        key, key_roll = jax.random.split(key)
        # 范围：-pi 到 +pi
        rand_roll = jax.random.uniform(key_roll, shape=(self.num_agents,), minval=-jnp.pi, maxval=jnp.pi)

        # # 使用 _quat_from_euler_bn 生成初始四元数
        # q_init = jax.vmap(_quat_from_euler_bn)(jnp.zeros_like(rand_pitch), rand_pitch, initial_heading)
        # [修改] 使用 rand_roll 生成四元数
        q_init = jax.vmap(_quat_from_euler_bn)(rand_roll, rand_pitch, initial_heading)
        
        # 更新飞机状态
        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading,
                # 显式更新欧拉角字段 (Dynamics通常会自动更新，但显式写更安全)
                roll=rand_roll,
                pitch=rand_pitch,
                vt=vt,
                vel_y=vt,
                q0=q_init[:, 0],
                q1=q_init[:, 1],
                q2=q_init[:, 2],
                q3=q_init[:, 3],
            )
        )
        
        # [修改] create 时传入 target_roll
        # extra_state 变成 4行: [heading, pitch, roll, vt]
        extra = jnp.stack([initial_heading, rand_pitch, rand_roll, vt], axis=0)
        state = Heading_Pitch_V_TaskState.create(state, extra_state=extra)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Heading_Pitch_V_TaskState:
        state = self._generate_formation(key, state, params)
        
        # 1. 随机航向
        # 随机生成初始航向角 (0 到 2π)
        key, key_heading = jax.random.split(key)
        # initial_heading = jnp.full((self.num_agents,), -jnp.pi)
        initial_heading = jax.random.uniform(
            key_heading, 
            shape=(self.num_agents,), 
            minval=0.0, 
            maxval=2.0 * jnp.pi
        )
        
        # 2. 初始速度
        # 设置初始速度
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_y = vt
        
        # 3. 随机俯仰 (带保护)
        # 修改后：加入随机 Pitch 初始化
        # 随机俯仰角变化量，考虑高度限制
        # 根据当前高度限制俯仰角变化范围
        current_altitude = state.plane_state.altitude
        # 如果高度接近上限，限制俯仰角为负值（向下）
        max_pitch = jnp.where(
            current_altitude > params.max_altitude - 1000,
            -params.max_pitch_increment * 0.5,  # 限制为负值，且幅度减半
            params.max_pitch_increment
        )
        # 如果高度接近下限，限制俯仰角为正值（向上）
        min_pitch = jnp.where(
            current_altitude < params.min_altitude + 1000,
            params.max_pitch_increment * 0.5,  # 限制为正值，且幅度减半
            -params.max_pitch_increment
        )
        key_pitch, key_heading = jax.random.split(key, 2)
        delta_pitch = jax.random.uniform(key_pitch, shape=(self.num_agents,), minval=min_pitch, maxval=max_pitch)
        
        # 计算新的俯仰角，并限制在安全范围内
        new_pitch = state.plane_state.pitch + delta_pitch
        # 限制最终俯仰角在安全范围内（通常在-89到+89度之间）
        safe_pitch_min = jnp.radians(-89.0)  # -89度
        safe_pitch_max = jnp.radians(89.0)   # +89度
        new_pitch = jnp.clip(new_pitch, safe_pitch_min, safe_pitch_max)
        # 随机生成 pitch (-89 ~ +89)
        rand_pitch = jax.random.uniform(key_pitch, shape=(self.num_agents,), minval=safe_pitch_min, maxval=safe_pitch_max)

        # 4. [新增] 随机滚转 (Roll)
        key, key_roll = jax.random.split(key)
        rand_roll = jax.random.uniform(key_roll, shape=(self.num_agents,), minval=-jnp.pi, maxval=jnp.pi)

        # [修改] 使用 rand_roll
        q_init = jax.vmap(_quat_from_euler_bn)(rand_roll, rand_pitch, initial_heading)

        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_y=vel_y,
                vt=vt,
                yaw=initial_heading,
                roll=rand_roll,
                pitch=rand_pitch,
                q0=q_init[:, 0],
                q1=q_init[:, 1],
                q2=q_init[:, 2],
                q3=q_init[:, 3],
            ),
            target_heading=initial_heading,
            target_pitch=rand_pitch,
            target_roll=rand_roll, # <--- [新增]
            target_vt=vt,
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: Heading_Pitch_V_TaskParams,
    ) -> Tuple[Heading_Pitch_V_TaskState, Dict[str, Any]]:
        """Task-specific step transition."""

        # [修改] 增加 key_roll
        key_heading, key_pitch, key_roll, key_vt_increment = jax.random.split(key, 4)
        # delta = self.increment_size[state.heading_turn_counts] # 渐进式增量系数
        delta = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=0.5, maxval=1.0)
         # 随机航向变化量(-π/3, π/3)
        delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        
        # 根据当前高度限制俯仰角变化范围
        current_altitude = state.plane_state.altitude
        # 如果高度接近上限，限制俯仰角为负值（向下）
        max_pitch = jnp.where(
            current_altitude > params.max_altitude - 1000,
            -params.max_pitch_increment * 0.5,  # 限制为负值，且幅度减半
            params.max_pitch_increment
        )
        # 如果高度接近下限，限制俯仰角为正值（向上）
        min_pitch = jnp.where(
            current_altitude < params.min_altitude + 1000,
            params.max_pitch_increment * 0.5,  # 限制为正值，且幅度减半
            -params.max_pitch_increment
        )
        # 随机俯仰角变化量，考虑高度限制
        delta_pitch = jax.random.uniform(key_pitch, shape=(self.num_agents,), minval=min_pitch, maxval=max_pitch)
        
        # 计算新的俯仰角，并限制在安全范围内
        new_pitch = state.plane_state.pitch + delta_pitch
        # 限制最终俯仰角在安全范围内（通常在-89到+89度之间）
        safe_pitch_min = jnp.radians(-89.0)  # -89度
        safe_pitch_max = jnp.radians(89.0)   # +89度
        new_pitch = jnp.clip(new_pitch, safe_pitch_min, safe_pitch_max)
        # 重新计算delta_pitch以确保符合限制
        delta_pitch = new_pitch - state.plane_state.pitch

        # [新增] 随机滚转变化量
        # Roll 不需要像 Pitch 那样做高度保护，它是自由的
        delta_roll = jax.random.uniform(key_roll, shape=(self.num_agents,), minval=-params.max_roll_increment, maxval=params.max_roll_increment)

        # 速度变化量(±100m/s)
        delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        target_heading = wrap_PI(state.plane_state.yaw + delta_heading * delta)
        target_pitch = wrap_PI(state.plane_state.pitch + delta_pitch * delta)

        # [新增] 目标滚转
        target_roll    = wrap_PI(state.plane_state.roll + delta_roll * delta)

        target_vt = state.plane_state.vt + delta_vt * delta

        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_roll=target_roll,  # <--- [新增] 更新
            target_vt=target_vt,
            last_check_time=state.time,
            heading_turn_counts=(state.heading_turn_counts + 1),
        )
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        info["heading_turn_counts"] = state.heading_turn_counts

        #================================================================#
        # ============================
        # 在 info 中记录“奖励被裁剪”的标志
        # ============================
        # 1) altitude 奖励：你在 altitude_reward_fn 里裁剪到 [-10, 10]，并且有“超过 10 表示将被裁剪”的监控变量（见 altitude_reward.py）
        #    参考：reward 输出裁剪阈值 [-10, 10] 与监控：cite__turn20file1
        ego_z_km = jnp.nan_to_num(state.plane_state.altitude / 1000.0, nan=0.0, posinf=1e6, neginf=-1e6)
        ego_vz_mh = jnp.nan_to_num(state.plane_state.vel_z / 340.0,    nan=0.0, posinf=1e6, neginf=-1e6)
        Kv = 0.2
        safe_alt = self.default_params.safe_altitude
        danger_alt = self.default_params.danger_altitude
        Pv = -jnp.clip(ego_vz_mh / Kv * (safe_alt - ego_z_km) / safe_alt, 0., 1.)
        # Pv = jax.lax.select(ego_z_km <= safe_alt, Pv, 0.0)
        Pv = jnp.where(ego_z_km <= safe_alt, Pv, jnp.zeros_like(Pv))
        PH = jnp.clip(ego_z_km / danger_alt, 0., 1.) - 1. - 1.
        # PH = jax.lax.select(ego_z_km <= danger_alt, PH, 0.0)
        PH = jnp.where(ego_z_km <= danger_alt, PH, jnp.zeros_like(PH))
        altitude_reward_raw = Pv + PH
        # “会被裁剪”的判定（与 altitude_reward_fn 里保持一致：绝对值>10 会被 clip 到 [-10,10]）
        altitude_reward_will_clip = jnp.abs(altitude_reward_raw) > 10.0

        # # 2) heading_pitch_V 奖励：你在 heading_pitch_V_reward_fn 里裁剪到 [0,1]
        # #    参考：输出 clip 到 [0,1] 与监控变量：cite__turn20file2
        # roll  = jnp.nan_to_num(state.plane_state.roll,  nan=0.0)
        # pitch = jnp.nan_to_num(state.plane_state.pitch, nan=0.0)
        # yaw   = jnp.nan_to_num(state.plane_state.yaw,   nan=0.0)
        # vt    = jnp.nan_to_num(state.plane_state.vt,    nan=0.0)

        # delta_heading = wrap_PI(yaw   - state.target_heading)
        # delta_pitch   = wrap_PI(pitch - state.target_pitch)
        # delta_vt      = jnp.nan_to_num(vt - state.target_vt, nan=0.0, posinf=1e6, neginf=-1e6)

        # # 与奖励函数保持同一尺度
        # heading_error_scale = jnp.pi / 72
        # pitch_error_scale   = jnp.pi / 72
        # roll_error_scale    = 0.35
        # speed_error_scale   = 24.0

        # # 按你的权重
        # w_heading = 0.4
        # w_pitch   = 0.3
        # w_roll    = 0.1
        # w_speed   = 0.2

        # # 这里计算“裁剪前”的 raw 值
        # heading_r = jnp.exp(-((jnp.clip(delta_heading, -jnp.pi, jnp.pi) / heading_error_scale) ** 2))
        # pitch_r   = jnp.exp(-((jnp.clip(delta_pitch,   -jnp.pi, jnp.pi) / pitch_error_scale) ** 2))
        # roll_r    = jnp.exp(-((jnp.clip(roll, -10.0, 10.0) / roll_error_scale) ** 2))
        # speed_r   = jnp.exp(-((jnp.clip(delta_vt, -1e3, 1e3)  / speed_error_scale) ** 2))

        # hpv_reward_raw = (heading_r**w_heading) * (pitch_r**w_pitch) * (roll_r**w_roll) * (speed_r**w_speed)
        # # “会被裁剪”的判定（>1 的部分会被 clip 到 1.0；一般不会>1，主要是数值噪声保护）
        # heading_pitch_V_reward_will_clip = hpv_reward_raw > 1.0

        # 2) heading_pitch_V 奖励（四元数）：口径与 reward_fn 保持一致
        q_curr = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)  # (B,4)

        def _theta_row(q_row, yh, ph, rh, vt, vt_tgt):
            q_err = _quat_err_nb(q_row, yh, ph, rh)
            w = jnp.clip(q_err[0], 0.0, 1.0)
            theta = 2.0 * jnp.arccos(w)
            theta_scale = jnp.pi / 36.0
            ori_r   = jnp.exp(- (theta/theta_scale)**2)
            speed_r = jnp.exp(- ((jnp.clip(jnp.nan_to_num(vt - vt_tgt, nan=0.0), -1e3, 1e3)/24.0)**2))
            return (ori_r**0.8) * (speed_r**0.2)

        hpv_reward_raw = jax.vmap(_theta_row, in_axes=(0,0,0,0,0,0))(
            q_curr, state.target_heading, state.target_pitch, state.target_roll, state.plane_state.vt, state.target_vt
        )
        heading_pitch_V_reward_will_clip = hpv_reward_raw > 1.0  # 理论不会 >1，这里只是保持同名监控键


        # 合并到 info（逐智能体布尔向量）
        info["clipped_altitude_reward_count"] = altitude_reward_will_clip.astype(jnp.float32)
        info["clipped_heading_pitch_V_reward_count"] = heading_pitch_V_reward_will_clip.astype(jnp.float32)
        info["clipped_any_reward_count"] = (altitude_reward_will_clip | heading_pitch_V_reward_will_clip).astype(jnp.float32)

        #================================================================#

        return state, info

    # @functools.partial(jax.jit, static_argnums=(0,))
    # def _get_obs(
    #     self,
    #     state: Heading_Pitch_V_TaskState,
    #     params: Heading_Pitch_V_TaskParams,
    # ) -> Dict[AgentName, chex.Array]:
    #     """
    #     Task-specific observation function to state.

    #     observation(dim 16):
    #         0. ego_delta_heading       (unit rad)
    #         1. ego_delta_pitch         (unit rad)  # 新增
    #         2. ego_delta_vt            (unit: mh)
    #         3. ego_altitude            (unit: 5km)
    #         4. ego_roll_sin
    #         5. ego_roll_cos
    #         6. ego_pitch_sin
    #         7. ego_pitch_cos
    #         8. ego_vt                  (unit: mh)
    #         9. ego_alpha_sin
    #         10. ego_alpha_cos
    #         11. ego_beta_sin
    #         12. ego_beta_cos
    #         13. ego_P                  (unit: rad/s)
    #         14. ego_Q                  (unit: rad/s)
    #         15. ego_R                  (unit: rad/s)
    #     """
    #     altitude = state.plane_state.altitude
    #     roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
    #     vt = state.plane_state.vt
    #     alpha = state.plane_state.alpha
    #     beta = state.plane_state.beta
    #     P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

    #     norm_delta_heading = wrap_PI((yaw - state.target_heading))
    #     norm_delta_pitch = wrap_PI((pitch - state.target_pitch))  # 新增
    #     norm_delta_vt = (vt - state.target_vt) / 340
    #     norm_altitude = altitude / 5000
    #     roll_sin = jnp.sin(roll)
    #     roll_cos = jnp.cos(roll)
    #     pitch_sin = jnp.sin(pitch)
    #     pitch_cos = jnp.cos(pitch)
    #     norm_vt = vt / 340
    #     alpha_sin = jnp.sin(alpha)
    #     alpha_cos = jnp.cos(alpha)
    #     beta_sin = jnp.sin(beta)
    #     beta_cos = jnp.cos(beta)
    #     obs = jnp.vstack((norm_delta_heading, norm_delta_pitch, norm_delta_vt,
    #                         norm_altitude, norm_vt,
    #                         roll_sin, roll_cos, pitch_sin, pitch_cos,
    #                         alpha_sin, alpha_cos, beta_sin, beta_cos,
    #                         P, Q, R))
    #     return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        16 维无奇异观测：
        [0:3]  q_err_vec (四元数误差向量部)
        [3]    (vt - target_vt)/340
        [4]    altitude / 5000
        [5]    vt / 340
        [6:9]  机体坐标系下的目标方向 v_b
        [9:12] P, Q, R
        [12:13]sin(alpha)
        [13:14]cos(alpha)
        [14:15]sin(beta)
        [15:16]cos(beta)
        """
        B = self.num_agents

        # 当前姿态 q_BN
        q_curr = jnp.stack([
            jnp.nan_to_num(state.plane_state.q0, nan=0.0),
            jnp.nan_to_num(state.plane_state.q1, nan=0.0),
            jnp.nan_to_num(state.plane_state.q2, nan=0.0),
            jnp.nan_to_num(state.plane_state.q3, nan=0.0),
        ], axis=1)  # (B,4)

        # 目标量
        yaw_t   = state.target_heading
        pitch_t = state.target_pitch
        roll_t  = state.target_roll # <--- [新增]
        vt_tgt  = state.target_vt

        # 批量误差四元数（w>=0），取向量部
        def _err_row(q_row, yh, ph, rh):
            q_err = _quat_err_nb(q_row, yh, ph, rh)
            return q_err
        q_err_batch = jax.vmap(_err_row, in_axes=(0,0,0,0))(q_curr, yaw_t, pitch_t, roll_t)  # (B,4)
        qv = jnp.clip(q_err_batch[:, 1:4], -1.0, 1.0)  # (B,3)

        # 机体系下的目标方向：由 (heading,pitch) 构一个 NED 单位向量再旋到 Body
        c_th, s_th = jnp.cos(yaw_t),   jnp.sin(yaw_t)
        c_ph, s_ph = jnp.cos(pitch_t), jnp.sin(pitch_t)
        v_n = jnp.stack([c_ph * c_th, c_ph * s_th, s_ph], axis=1)  # (B,3)
        v_b = jax.vmap(_rotate_ned_to_body, in_axes=(0,0))(q_curr, v_n)
        v_b = jnp.clip(v_b, -1.0, 1.0)  # (B,3)

        # 其它飞参
        altitude = state.plane_state.altitude         # (B,)
        vt       = state.plane_state.vt
        alpha    = state.plane_state.alpha
        beta     = state.plane_state.beta
        P, Q, R  = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_dvt = (vt - vt_tgt) / 340.0
        norm_alt = altitude / 5000.0
        norm_vt  = vt / 340.0

        alpha_sin, alpha_cos = jnp.sin(alpha), jnp.cos(alpha)
        beta_sin,  beta_cos  = jnp.sin(beta),  jnp.cos(beta)

        # (feat, B)
        obs_mat = jnp.stack([
            qv[:,0], qv[:,1], qv[:,2],       # 0-2
            norm_dvt,                        # 3
            norm_alt,                        # 4
            norm_vt,                         # 5
            v_b[:,0], v_b[:,1], v_b[:,2],    # 6-8
            P, Q, R,                          # 9-11
            alpha_sin, alpha_cos,            # 12-13
            beta_sin,  beta_cos              # 14-15
        ], axis=0)

        # 数值保护与夹限
        low  = jnp.array([-1., -1., -1., -2., 0., 0., -1., -1., -1., -10., -10., -10., -1., -1., -1., -1.]).reshape(-1,1)
        high = jnp.array([ 1.,  1.,  1.,  2., 5., 2.,  1.,  1.,  1.,  10.,  10.,  10.,  1.,  1.,  1.,  1.]).reshape(-1,1)
        obs_mat = jnp.clip(jnp.nan_to_num(obs_mat, nan=0.0, posinf=0.0, neginf=0.0), low, high)

        return {agent: obs_mat[:, i] for i, agent in enumerate(self.agents)}

    # @functools.partial(jax.jit, static_argnums=(0,))
    # def _get_obs(
    #     self,
    #     state: Heading_Pitch_V_TaskState,
    #     params: Heading_Pitch_V_TaskParams,
    # ) -> Dict[AgentName, chex.Array]:
    #     """
    #     28 维观测（扩展版）：
    #     [0:3]   q_err_vec (四元数误差向量部)
    #     [3]     (vt - target_vt)/340
    #     [4]     altitude / 5000
    #     [5]     vt / 340
    #     [6:9]   机体坐标系下的目标方向 v_b
    #     [9:12]  P, Q, R (角速度)
    #     [12:14] sin/cos(alpha)
    #     [14:16] sin/cos(beta)
    #     --- 新增 ---
    #     [16:18] north/10000, east/10000 (位置)
    #     [18:21] vel_x/340, vel_y/340, vel_z/340 (速度分量)
    #     [21:24] ax, ay, az (过载，单位 g)
    #     [24:28] T/T_max, el/45, ail/45, rud/45 (控制状态归一化)
    #     """
    #     B = self.num_agents

    #     # 当前姿态 q_BN
    #     q_curr = jnp.stack([
    #         jnp.nan_to_num(state.plane_state.q0, nan=0.0),
    #         jnp.nan_to_num(state.plane_state.q1, nan=0.0),
    #         jnp.nan_to_num(state.plane_state.q2, nan=0.0),
    #         jnp.nan_to_num(state.plane_state.q3, nan=0.0),
    #     ], axis=1)  # (B,4)

    #     # 目标量
    #     yaw_t   = state.target_heading
    #     pitch_t = state.target_pitch
    #     roll_t  = state.target_roll
    #     vt_tgt  = state.target_vt

    #     # 批量误差四元数（w>=0），取向量部
    #     def _err_row(q_row, yh, ph, rh):
    #         q_err = _quat_err_nb(q_row, yh, ph, rh)
    #         return q_err
    #     q_err_batch = jax.vmap(_err_row, in_axes=(0,0,0,0))(q_curr, yaw_t, pitch_t, roll_t)  # (B,4)
    #     qv = jnp.clip(q_err_batch[:, 1:4], -1.0, 1.0)  # (B,3)

    #     # 机体系下的目标方向
    #     c_th, s_th = jnp.cos(yaw_t),   jnp.sin(yaw_t)
    #     c_ph, s_ph = jnp.cos(pitch_t), jnp.sin(pitch_t)
    #     v_n = jnp.stack([c_ph * c_th, c_ph * s_th, s_ph], axis=1)  # (B,3)
    #     v_b = jax.vmap(_rotate_ned_to_body, in_axes=(0,0))(q_curr, v_n)
    #     v_b = jnp.clip(v_b, -1.0, 1.0)  # (B,3)

    #     # 飞行参数
    #     altitude = state.plane_state.altitude
    #     vt       = state.plane_state.vt
    #     alpha    = state.plane_state.alpha
    #     beta     = state.plane_state.beta
    #     P, Q, R  = state.plane_state.P, state.plane_state.Q, state.plane_state.R

    #     norm_dvt = (vt - vt_tgt) / 340.0
    #     norm_alt = altitude / 5000.0
    #     norm_vt  = vt / 340.0

    #     alpha_sin, alpha_cos = jnp.sin(alpha), jnp.cos(alpha)
    #     beta_sin,  beta_cos  = jnp.sin(beta),  jnp.cos(beta)

    #     # ========== 新增：位置、速度分量、过载、控制状态 ==========
    #     # 位置（归一化到 ±1 范围，假设地图 ±10km）
    #     north = state.plane_state.north
    #     east  = state.plane_state.east
    #     norm_north = north / 10000.0
    #     norm_east  = east / 10000.0

    #     # 速度分量（NED 坐标系，归一化到马赫数）
    #     vel_x = state.plane_state.vel_x
    #     vel_y = state.plane_state.vel_y
    #     vel_z = state.plane_state.vel_z
    #     norm_vel_x = vel_x / 340.0
    #     norm_vel_y = vel_y / 340.0
    #     norm_vel_z = vel_z / 340.0

    #     # 过载（单位 g，已经是无量纲的）
    #     ax = state.plane_state.ax
    #     ay = state.plane_state.ay
    #     az = state.plane_state.az

    #     # 控制状态（归一化）
    #     T   = state.plane_state.T
    #     el  = state.plane_state.el
    #     ail = state.plane_state.ail
    #     rud = state.plane_state.rud
    #     T_max = 1.0 * 0.225 * 76300 / 0.3048  # ≈ 56316
    #     norm_T   = T / T_max            # 范围 [0, 1]
    #     norm_el  = el / 45.0            # 范围 [-1, 1]
    #     norm_ail = ail / 45.0           # 范围 [-1, 1]
    #     norm_rud = rud / 45.0           # 范围 [-1, 1]
    #     # ==========================================================

    #     # 构建观测矩阵 (feat, B)
    #     obs_mat = jnp.stack([
    #         qv[:,0], qv[:,1], qv[:,2],       # 0-2: 姿态误差
    #         norm_dvt,                        # 3: 速度误差
    #         norm_alt,                        # 4: 高度
    #         norm_vt,                         # 5: 速度
    #         v_b[:,0], v_b[:,1], v_b[:,2],    # 6-8: 目标方向
    #         P, Q, R,                          # 9-11: 角速度
    #         alpha_sin, alpha_cos,            # 12-13: 攻角
    #         beta_sin,  beta_cos,             # 14-15: 侧滑角
    #         # --- 新增 ---
    #         norm_north, norm_east,           # 16-17: 归一化后的位置
    #         norm_vel_x, norm_vel_y, norm_vel_z,  # 18-20: 归一化后的速度分量
    #         ax, ay, az,                      # 21-23: 过载
    #         norm_T, norm_el, norm_ail, norm_rud  # 24-27: 归一化后的控制状态
    #     ], axis=0)

    #     # 数值保护与夹限（扩展到 28 维）
    #     low  = jnp.array([
    #         -1., -1., -1.,   # qv
    #         -2.,             # dvt
    #         0.,              # alt
    #         0.,              # vt
    #         -1., -1., -1.,   # v_b
    #         -10., -10., -10.,# P,Q,R
    #         -1., -1.,        # alpha sin/cos
    #         -1., -1.,        # beta sin/cos
    #         -10., -10.,      # north, east
    #         -2., -2., -2.,   # vel_x, vel_y, vel_z
    #         -15., -15., -15.,# ax, ay, az (过载范围约 ±9g，留余量)
    #         0., -1., -1., -1.# T, el, ail, rud
    #     ]).reshape(-1,1)
        
    #     high = jnp.array([
    #         1., 1., 1.,      # qv
    #         2.,              # dvt
    #         5.,              # alt
    #         2.,              # vt
    #         1., 1., 1.,      # v_b
    #         10., 10., 10.,   # P,Q,R
    #         1., 1.,          # alpha sin/cos
    #         1., 1.,          # beta sin/cos
    #         10., 10.,        # north, east
    #         2., 2., 2.,      # vel_x, vel_y, vel_z
    #         15., 15., 15.,   # ax, ay, az
    #         1., 1., 1., 1.   # T, el, ail, rud
    #     ]).reshape(-1,1)
        
    #     obs_mat = jnp.clip(jnp.nan_to_num(obs_mat, nan=0.0, posinf=0.0, neginf=0.0), low, high)

    #     return {agent: obs_mat[:, i] for i, agent in enumerate(self.agents)}
    
    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: Heading_Pitch_V_TaskState,
            params: Heading_Pitch_V_TaskParams,
        ) -> Heading_Pitch_V_TaskState:

        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center =  team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
        initial_heading = jnp.full((self.num_agents,), jnp.pi/2)
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2],
            yaw=initial_heading,
        ))
        return state

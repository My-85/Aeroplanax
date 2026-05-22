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
    heading_pitch_V_reward_euler_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_V_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class Heading_Pitch_V_TaskState(EnvState):
    target_heading: ArrayLike
    target_pitch: ArrayLike
    target_roll: ArrayLike
    target_vt: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=extra_state[0],
            target_pitch=extra_state[1],
            target_roll=extra_state[2],
            target_vt=extra_state[3],
            last_check_time=env_state.time,
            heading_turn_counts=jnp.zeros_like(env_state.plane_state.north, dtype=jnp.int32),
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
    max_vt: float = 360.0
    min_vt: float = 120.0
    max_heading_increment: float = jnp.pi/2  # 最大航向变化量(90°)
    max_pitch_increment: float = jnp.pi/6  # 最大俯仰角变化量(30°)
    max_altitude_increment: float = 2100.0
    max_velocities_u_increment: float = 50.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000 # 编队最小安全间距


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
            functools.partial(heading_pitch_V_reward_euler_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=0),
        ]

        # 与 reward_functions 一一对应，表示这些奖励是否做势能差分
        # 这里全部设为 False 即可
        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_V_fn,
        ]

        # 课程学习：
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)
        # 前5个元素是 [0.2, 0.4, 0.6, 0.8, 1.0]
        # 后10个元素是 [1.0] 重复10次
        # 该数组用于控制航向/俯仰/速度变化量的增量系数
        # 每次 heading_turn_counts 增加时，会按索引取对应的系数值进行缩放
        # 前5次任务切换时增量系数逐步增大（0.2→1.0），后续保持1.0不变

    def _get_obs_size(self) -> int:
        return 22  # 17 state (incl delta_roll) + 5 previous action

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

        # 随机航向角 (0 到 2π)，保持初始方向的多样性
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(
            key_heading,
            shape=(self.num_agents,),
            minval=0.0,
            maxval=2.0 * jnp.pi
        )

        # 配平巡航速度 (trimmed cruise, 250 m/s ≈ Mach 0.74 at 5000m)
        vt = jnp.full((self.num_agents,), 250.0)
        
        # 根据随机航向角计算对应的四元数
        half_heading = initial_heading / 2.0
        q0 = -jnp.cos(half_heading)  # cos(θ/2)
        q1 = jnp.zeros((self.num_agents,))
        q2 = jnp.zeros((self.num_agents,))
        q3 = jnp.sin(half_heading)  # sin(θ/2)
        
        # 更新飞机状态
        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=initial_heading,
                vt=vt,
                vel_y=vt,
                q0=q0,
                q1=q1,
                q2=q2,
                q3=q3,
            )
        )
        
        state = Heading_Pitch_V_TaskState.create(state, extra_state=jnp.zeros((4, self.num_agents)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Heading_Pitch_V_TaskState:
        state = self._generate_formation(key, state, params)

        # 配平初始化: 固定巡航高度 + 巡航速度，航向随机
        state = state.replace(
            plane_state=state.plane_state.replace(
                altitude=jnp.full((self.num_agents,), 5000.0),
            )
        )

        # 随机航向角 (0 到 2π)
        key, key_heading = jax.random.split(key)
        initial_heading = jax.random.uniform(
            key_heading,
            shape=(self.num_agents,),
            minval=0.0,
            maxval=2.0 * jnp.pi
        )

        # 配平巡航速度
        vt = jnp.full((self.num_agents,), 250.0)
        vel_y = vt

        # 根据随机航向角计算对应的四元数
        half_heading = initial_heading / 2.0
        q0 = -jnp.cos(half_heading)  # cos(θ/2)
        q1 = jnp.zeros((self.num_agents,))
        q2 = jnp.zeros((self.num_agents,))
        q3 = jnp.sin(half_heading)  # sin(θ/2)

        # 首个目标 = 当前配平状态（零误差起步，加速早期学习）
        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_y=vel_y,
                vt=vt,
                yaw=initial_heading,
                q0=q0,
                q1=q1,
                q2=q2,
                q3=q3,
            ),
            target_heading=initial_heading,
            target_pitch=jnp.zeros((self.num_agents,)),
            target_roll=jnp.zeros((self.num_agents,)),
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
        # TODO: only fit single agent

        # ── 全空间绝对目标生成 ──
        # heading/pitch/vt: 独立于当前状态，覆盖全域
        # roll: 60% 平飞 (0°) + 40% 任意滚转 (包括倒飞)
        key_h, key_p, key_r, key_v = jax.random.split(key, 4)
        target_heading = jax.random.uniform(key_h, shape=(self.num_agents,), minval=-jnp.pi, maxval=jnp.pi)
        target_pitch   = jax.random.uniform(key_p, shape=(self.num_agents,),
                                            minval=jnp.deg2rad(-89.0), maxval=jnp.deg2rad(89.0))
        target_vt      = jax.random.uniform(key_v, shape=(self.num_agents,),
                                            minval=params.min_vt, maxval=params.max_vt)

        key_roll_tier, key_roll_val = jax.random.split(key_r)
        is_random_roll = jax.random.bernoulli(key_roll_tier, 0.4, shape=(self.num_agents,))
        target_roll = jnp.where(is_random_roll,
            jax.random.uniform(key_roll_val, shape=(self.num_agents,), minval=-jnp.pi, maxval=jnp.pi),
            jnp.zeros((self.num_agents,)))

        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_roll=target_roll,
            target_vt=target_vt,
            last_check_time=state.time,
            heading_turn_counts=(state.heading_turn_counts + 1),  # monitoring counter
        )
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        info["heading_turn_counts"] = state.heading_turn_counts

        #================================================================#
        # 在 info 中记录每个 reward 分量的实际值（供 wandb / 终端打印）
        # pre_rewards[i, agent_id] 对应 reward_functions[i]
        #   [0] = heading_pitch_V_reward_euler_fn  → attitude + speed
        #   [1] = altitude_reward_fn                → altitude penalty
        #   [2] = event_driven_reward_fn            → crash penalty (-200/0)
        pre = state.pre_rewards  # shape (3, num_agents)
        info["r_attitude_v"] = pre[0]   # per-agent attitude+speed reward
        info["r_altitude"]   = pre[1]   # per-agent altitude penalty
        info["r_crash"]      = pre[2]   # per-agent crash penalty

        # 同时记录各通道的原始误差（便于诊断）
        roll  = jnp.nan_to_num(state.plane_state.roll,  nan=0.0)
        pitch = jnp.nan_to_num(state.plane_state.pitch, nan=0.0)
        yaw   = jnp.nan_to_num(state.plane_state.yaw,   nan=0.0)
        vt    = jnp.nan_to_num(state.plane_state.vt,    nan=0.0)
        tgt_heading = jnp.nan_to_num(state.target_heading, nan=0.0)
        tgt_pitch   = jnp.nan_to_num(state.target_pitch,   nan=0.0)
        tgt_roll    = jnp.nan_to_num(state.target_roll,    nan=0.0)
        tgt_vt      = jnp.nan_to_num(state.target_vt,      nan=0.0)
        delta_heading = wrap_PI(yaw - tgt_heading)
        delta_pitch   = wrap_PI(pitch - tgt_pitch)
        delta_roll    = wrap_PI(roll - tgt_roll)
        delta_vt      = jnp.nan_to_num(vt - tgt_vt, nan=0.0)
        info["dbg_heading_err_deg"] = jnp.abs(delta_heading) * 180.0 / jnp.pi
        info["dbg_pitch_err_deg"]   = jnp.abs(delta_pitch)   * 180.0 / jnp.pi
        info["dbg_roll_err_deg"]    = jnp.abs(delta_roll)    * 180.0 / jnp.pi
        info["dbg_speed_err_ms"]    = jnp.abs(delta_vt)
        info["dbg_alt_km"]          = state.plane_state.altitude / 1000.0
        info["dbg_vt_ms"]           = vt
        #================================================================#

        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: Heading_Pitch_V_TaskState,
        params: Heading_Pitch_V_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.

        observation(dim 22):
             0. ego_delta_heading       (unit rad)
             1. ego_delta_pitch         (unit rad)
             2. ego_delta_roll          (unit rad)
             3. ego_delta_vt            (unit: 340 m/s)
             4. ego_altitude            (unit: 5km)
             5. ego_vt                  (unit: 340 m/s)
             6. ego_roll_sin
             7. ego_roll_cos
             8. ego_pitch_sin
             9. ego_pitch_cos
            10. ego_alpha_sin
            11. ego_alpha_cos
            12. ego_beta_sin
            13. ego_beta_cos
            14. ego_P                  (unit: rad/s)
            15. ego_Q                  (unit: rad/s)
            16. ego_R                  (unit: rad/s)
            17. prev_throttle           (0~1)
            18. prev_elevator           (-1~1)
            19. prev_aileron            (-1~1)
            20. prev_rudder             (-1~1)
            21. prev_speed_brake        (0~1)
        """
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_pitch   = wrap_PI((pitch - state.target_pitch))
        norm_delta_roll    = wrap_PI((roll - state.target_roll))
        norm_delta_vt = (vt - state.target_vt) / 340
        norm_altitude = altitude / 5000
        norm_vt = vt / 340
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)

        # Previous action — reduces GRU burden for actuator dynamics
        cs = state.control_state
        prev_thr = jnp.nan_to_num(cs.throttle,     nan=0.0)
        prev_el  = jnp.nan_to_num(cs.elevator,     nan=0.0)
        prev_ail = jnp.nan_to_num(cs.aileron,      nan=0.0)
        prev_rud = jnp.nan_to_num(cs.rudder,       nan=0.0)
        prev_sb  = jnp.nan_to_num(cs.speed_brake,  nan=0.0)

        obs = jnp.vstack((norm_delta_heading, norm_delta_pitch, norm_delta_roll, norm_delta_vt,
                            norm_altitude, norm_vt,
                            roll_sin, roll_cos, pitch_sin, pitch_cos,
                            alpha_sin, alpha_cos, beta_sin, beta_cos,
                            P, Q, R,
                            prev_thr, prev_el, prev_ail, prev_rud, prev_sb))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
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

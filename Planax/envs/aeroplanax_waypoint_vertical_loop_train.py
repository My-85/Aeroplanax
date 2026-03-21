# Planax/envs/aeroplanax_waypoint_vertical_loop_simple.py
import functools
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from .aeroplanax import AgentName, AgentID, EnvState, EnvParams, AeroPlanaxEnv
from .core.simulators import fighterplane


@struct.dataclass(frozen=True)
class VerticalLoopParams(EnvParams):
    # 基本
    max_steps: int = 3000
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    action_type: int = 1  # 1=离散 4 通道
    # 初始包线
    max_altitude: float = 20000.0
    min_altitude: float = 8000.0
    max_vt: float = 340.0
    min_vt: float = 100.0

    # 筋斗参数（仅此一种任务，无分支）
    loop_radius: float = 6000.0
    loop_points_per_circle: int = 240
    loop_forward_north: float = 0.0           # 每圈圆心前推（m），0 表示原地画圈
    loop_target_vt: float = 230.0
    loop_phase0_deg: float = 180.0            # 首点在“正前方、同高”
    loop_direction: int = -1                  # -1 顺时针（φ递减），+1 逆时针
    success_after_loops: int = 1              # 新增：完成多少圈判定成功

    # 自适应前视与俯仰限幅
    lookahead_base_pts: int = 8
    lookahead_gain_pts: int = 24
    pitch_limit_deg_up: float = 55.0
    pitch_limit_deg_down: float = 70.0        # 向下更宽松

    # 航点达成
    reach_radius_init: float = 800.0
    reach_radius_decay: float = 1.0
    max_waypoints: int = 10**9                # 不以达成个数终止

    # 训练接口（不使用内置baseline）
    use_internal_baseline: bool = struct.field(pytree_node=False, default=False)




@struct.dataclass
class VerticalLoopState(EnvState):
    waypoint: jnp.ndarray           # (3,) 当前目标航点 [n,e,a]
    reached: jnp.ndarray            # 已达成计数
    reach_radius: jnp.ndarray       # 当前判定半径
    loop_center_n: jnp.ndarray
    loop_center_e: jnp.ndarray
    loop_center_alt: jnp.ndarray
    loop_idx: jnp.ndarray           # 当前相位索引（0..points-1）


def _wrap_pi(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


def _bearing(dn, de):
    return jnp.arctan2(de, dn)


def _desired_pitch(d_alt, h_dist):
    return jnp.arctan2(d_alt, jnp.maximum(h_dist, 1e-6))


class VerticalLoopEnv(AeroPlanaxEnv[VerticalLoopState, VerticalLoopParams]):
    def __init__(self, env_params: Optional[VerticalLoopParams] = None):
        super().__init__(env_params)
        self._default_params = env_params or VerticalLoopParams()

        # spaces（16维与训练对齐）
        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: spaces.Box(-jnp.inf, jnp.inf, (16,), dtype=jnp.float32) for agent in self.agents
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: spaces.Dict({
                "throttle": spaces.Discrete(31),
                "elevator": spaces.Discrete(41),
                "aileron":  spaces.Discrete(41),
                "rudder":   spaces.Discrete(41),
            }) for agent in self.agents
        }

        # 奖励
        self.reward_functions = [
            functools.partial(self._r_dist, scale=1.0),
            functools.partial(self._r_align, scale=0.3),
            functools.partial(self._r_circle_path, scale=0.2),     # 新增：圆径向误差
            functools.partial(self._r_speed_penalty, scale=-0.05),
            functools.partial(self._r_overload_penalty, scale=-0.1),
            functools.partial(self._r_reach_bonus, bonus=2.0),
        ]
        self.is_potential = [False, False, False, False, False, False]

        # 终止
        self.termination_conditions = [
            self._term_success,
            self._term_timeout,
            self._term_overspeed,   # 新增
            self._term_crashed,
        ]

    def _get_obs_size(self) -> int:
        return 16

    @property
    def default_params(self) -> VerticalLoopParams:
        return self._default_params

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key, params: VerticalLoopParams) -> VerticalLoopState:
        s = super()._init_state(key, params)

        # 随机初值（vt/alt）
        key, kv, ka = jax.random.split(key, 3)
        vt0 = jax.random.uniform(kv, (1,), minval=params.min_vt, maxval=params.max_vt)
        alt0 = jax.random.uniform(ka, (1,), minval=params.min_altitude, maxval=params.max_altitude)
        s = s.replace(plane_state=s.plane_state.replace(vt=vt0, altitude=alt0))

        # 圆心在前方 R，East 固定当前
        n0 = s.plane_state.north[0]
        e0 = s.plane_state.east[0]
        a0 = s.plane_state.altitude[0]
        R = params.loop_radius
        c_n, c_e, c_a = n0 + R, e0, a0

        # 首点相位
        dphi = 2 * jnp.pi / params.loop_points_per_circle
        dir_ = jnp.sign(jnp.asarray(params.loop_direction, jnp.float32))
        phi0 = jnp.deg2rad(params.loop_phase0_deg)
        idx0 = jnp.array(0, jnp.int32)
        phi = phi0 + dir_ * idx0.astype(jnp.float32) * dphi

        n1 = c_n - R * jnp.cos(phi)
        e1 = c_e
        a1 = c_a + R * jnp.sin(phi)
        wp = jnp.array([n1, e1, a1])

        return VerticalLoopState(
            plane_state=s.plane_state,
            missile_state=s.missile_state,
            control_state=s.control_state,
            pre_rewards=jnp.zeros((len(self.reward_functions), self.num_agents)),
            done=False,
            success=False,
            time=0,

            waypoint=wp,
            reached=jnp.array(0),
            reach_radius=jnp.array(params.reach_radius_init),
            loop_center_n=jnp.array(c_n),
            loop_center_e=jnp.array(c_e),
            loop_center_alt=jnp.array(c_a),
            loop_idx=idx0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(self, key, state: VerticalLoopState, params: VerticalLoopParams) -> VerticalLoopState:
        # 与 _init_state 相同（无课变）
        n0 = state.plane_state.north[0]
        e0 = state.plane_state.east[0]
        a0 = state.plane_state.altitude[0]
        R = params.loop_radius
        c_n, c_e, c_a = n0 + R, e0, a0
        dphi = 2 * jnp.pi / params.loop_points_per_circle
        dir_ = jnp.sign(jnp.asarray(params.loop_direction, jnp.float32))
        phi0 = jnp.deg2rad(params.loop_phase0_deg)
        idx0 = jnp.array(0, jnp.int32)
        phi = phi0 + dir_ * idx0.astype(jnp.float32) * dphi
        n1 = c_n - R * jnp.cos(phi)
        e1 = c_e
        a1 = c_a + R * jnp.sin(phi)
        wp = jnp.array([n1, e1, a1])
        return state.replace(
            waypoint=wp,
            reached=jnp.array(0),
            reach_radius=jnp.array(params.reach_radius_init),
            loop_center_n=jnp.array(c_n),
            loop_center_e=jnp.array(c_e),
            loop_center_alt=jnp.array(c_a),
            loop_idx=idx0,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key,
        state: VerticalLoopState,
        info: Dict[str, jnp.ndarray],
        action: Dict[AgentName, jnp.ndarray],
        params: VerticalLoopParams,
    ) -> Tuple[VerticalLoopState, Dict[str, jnp.ndarray]]:
        # 目标姿态：自适应前视 + 切向引导
        pn, pe, pa = state.plane_state.north[0], state.plane_state.east[0], state.plane_state.altitude[0]
        R = params.loop_radius
        dphi = 2 * jnp.pi / params.loop_points_per_circle
        dir_ = jnp.sign(jnp.asarray(params.loop_direction, jnp.float32))
        phi0 = jnp.deg2rad(params.loop_phase0_deg)
        phi_cur = phi0 + dir_ * state.loop_idx.astype(jnp.float32) * dphi

        boost = 1.0 - jnp.abs(jnp.cos(phi_cur))
        L = jnp.asarray(params.lookahead_base_pts, jnp.float32) + jnp.asarray(params.lookahead_gain_pts, jnp.float32) * boost
        L = jnp.clip(L, jnp.asarray(params.lookahead_base_pts, jnp.float32),
                     jnp.asarray(params.loop_points_per_circle, jnp.float32) * 0.5)
        phi_L = phi_cur + dir_ * L * dphi

        nL = state.loop_center_n - R * jnp.cos(phi_L)
        eL = state.loop_center_e
        aL = state.loop_center_alt + R * jnp.sin(phi_L)

        phi_next = phi_L + dir_ * dphi
        nT = state.loop_center_n - R * jnp.cos(phi_next)
        eT = state.loop_center_e
        aT = state.loop_center_alt + R * jnp.sin(phi_next)
        tn, te, ta = (nT - nL), (eT - eL), (aT - aL)
        hd = jnp.sqrt(jnp.maximum(tn * tn + te * te, 1e-6))

        desired_heading = _bearing(tn, te)
        desired_pitch = _desired_pitch(ta, hd)
        desired_pitch = jnp.clip(
            desired_pitch,
            -jnp.deg2rad(params.pitch_limit_deg_down),
            jnp.deg2rad(params.pitch_limit_deg_up),
        )

        # 航点达成（针对“当前 wp”，而非前视点）
        dn, de, da = state.waypoint[0] - pn, state.waypoint[1] - pe, state.waypoint[2] - pa
        hdist = jnp.sqrt(dn * dn + de * de)
        dist3d = jnp.sqrt(hdist * hdist + da * da)
        reached_now = dist3d <= state.reach_radius

        info['dist_to_wp'] = dist3d
        info['hdist_to_wp'] = hdist
        info['dbg_desired_pitch'] = desired_pitch
        info['dbg_desired_heading'] = desired_heading
        info['dbg_target_vt'] = jnp.asarray(params.loop_target_vt, jnp.float32)
        info['reach_radius'] = state.reach_radius
        info['reached_count'] = state.reached
        info['loops_completed'] = state.reached // jnp.asarray(params.loop_points_per_circle, jnp.int32)

        def on_reach(_):
            new_idx = state.loop_idx + 1
            full = new_idx >= jnp.asarray(params.loop_points_per_circle, jnp.int32)
            new_idx = jnp.where(full, jnp.array(0, jnp.int32), new_idx)
            c_n = jnp.where(full, state.loop_center_n + jnp.asarray(params.loop_forward_north, jnp.float32), state.loop_center_n)
            c_e = state.loop_center_e
            c_a = state.loop_center_alt
            phi = phi0 + dir_ * new_idx.astype(jnp.float32) * dphi
            n_next = c_n - R * jnp.cos(phi)
            e_next = c_e
            a_next = c_a + R * jnp.sin(phi)
            wp = jnp.array([n_next, e_next, a_next])
            reach_radius = jnp.maximum(200.0, state.reach_radius * params.reach_radius_decay)
            return state.replace(
                waypoint=wp,
                reach_radius=reach_radius,
                reached=state.reached + 1,
                loop_center_n=c_n,
                loop_center_e=c_e,
                loop_center_alt=c_a,
                loop_idx=new_idx,
            )

        def on_keep(_):
            return state

        state = jax.lax.cond(reached_now, on_reach, on_keep, operand=None)
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: VerticalLoopState, params: VerticalLoopParams) -> Dict[AgentName, jnp.ndarray]:
        dn = state.waypoint[0] - state.plane_state.north
        de = state.waypoint[1] - state.plane_state.east
        da = state.waypoint[2] - state.plane_state.altitude
        hdist = jnp.sqrt(jnp.maximum(dn * dn + de * de, 1e-6))
        desired_heading = _bearing(dn, de)
        desired_pitch = _desired_pitch(da, hdist)

        vt = state.plane_state.vt
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        alpha, beta = state.plane_state.alpha, state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        obs = jnp.vstack((
            _wrap_pi(yaw - desired_heading),
            _wrap_pi(pitch - desired_pitch),
            (vt - params.loop_target_vt) / 340.0,
            altitude / 5000.0,
            vt / 340.0,
            jnp.sin(roll),  jnp.cos(roll),
            jnp.sin(pitch), jnp.cos(pitch),
            jnp.sin(alpha), jnp.cos(alpha),
            jnp.sin(beta),  jnp.cos(beta),
            P, Q, R
        ))
        # 先做 NaN/Inf 清洗，再 clip（关键）
        obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = jnp.clip(
            obs,
            jnp.array([ -jnp.pi, -jnp.pi, -2.0, 0.0, 0.0, -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1., -10.,-10.,-10. ]).reshape(-1,1),
            jnp.array([  jnp.pi,  jnp.pi,  2.0, 5.0, 2.0,  1., 1., 1., 1., 1., 1., 1., 1.,  10., 10., 10. ]).reshape(-1,1),
        )
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    # 终止

    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_success(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID):
        loops_done = state.reached // jnp.asarray(params.loop_points_per_circle, jnp.int32)
        succ = loops_done >= jnp.asarray(params.success_after_loops, jnp.int32)
        return succ, jnp.array(True)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_timeout(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID):
        limit = (jnp.asarray(params.max_steps, jnp.float32)
                 * jnp.asarray(params.sim_freq, jnp.float32)
                 / jnp.asarray(params.agent_interaction_steps, jnp.float32))
        done = jnp.asarray(state.time, jnp.float32) >= limit
        return done, jnp.array(False)

    # 终止
    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_crashed(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID):
        crashed = state.plane_state.status[agent_id] == 2
        return crashed, jnp.array(False)

    # 新增：超速硬终止（很宽，防止数值爆掉）
    @functools.partial(jax.jit, static_argnums=(0,))
    def _term_overspeed(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID):
        done = state.plane_state.vt[agent_id] > (params.max_vt * 2.0)
        return done, jnp.array(False)

    # 奖励
    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_dist(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID, scale=1.0):
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        dist = jnp.sqrt(dn*dn + de*de + da*da)
        return scale * (-dist / 10000.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_align(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID, scale=0.3):
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        hdist = jnp.sqrt(jnp.maximum(dn*dn + de*de, 1e-6))
        desired_heading = _bearing(dn, de)
        desired_pitch = _desired_pitch(da, hdist)
        yaw, pitch = state.plane_state.yaw[0], state.plane_state.pitch[0]
        ah = jnp.exp(-((_wrap_pi(yaw - desired_heading)) / (jnp.pi/8))**2)
        ap = jnp.exp(-((_wrap_pi(pitch - desired_pitch)) / (jnp.pi/12))**2)
        return scale * (0.5*ah + 0.5*ap)

    # 超速惩罚
    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_speed_penalty(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID, scale=-0.05):
        vt = state.plane_state.vt[0]
        penalty = jnp.clip(vt - params.loop_target_vt, 0.0)**2 / (jnp.maximum(params.max_vt, 1.0)**2)
        return scale * penalty

    # Nz过载惩罚
    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_overload_penalty(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID, scale=-0.1, nz_lim=8.0):
        nz = jnp.abs(state.plane_state.az[0])
        return scale * jnp.clip(nz - nz_lim, 0.0)**2

    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_reach_bonus(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID, bonus=2.0):
        dn = state.waypoint[0] - state.plane_state.north[0]
        de = state.waypoint[1] - state.plane_state.east[0]
        da = state.waypoint[2] - state.plane_state.altitude[0]
        dist = jnp.sqrt(dn*dn + de*de + da*da)
        return jnp.where(dist <= state.reach_radius, bonus, 0.0)

    # 圆径向误差塑形（不和俯仰重复）：半径越接近 R 奖励越大
    @functools.partial(jax.jit, static_argnums=(0,))
    def _r_circle_path(self, state: VerticalLoopState, params: VerticalLoopParams, agent_id: AgentID, scale=0.2):
        cn, ca = state.loop_center_n, state.loop_center_alt
        n, a = state.plane_state.north[0], state.plane_state.altitude[0]
        R = jnp.asarray(params.loop_radius, jnp.float32)
        r = jnp.sqrt(jnp.maximum((cn - n) ** 2 + (a - ca) ** 2, 1e-6))
        err = jnp.abs(r - R) / jnp.maximum(R, 1.0)
        return scale * (-err)


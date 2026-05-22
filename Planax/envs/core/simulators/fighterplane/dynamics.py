import jax
import jax.numpy as jnp
from flax import struct
from . import aero_data as hifi_F16
from ...base_dataclass import BasePlaneState, BaseControlState


@struct.dataclass
class FighterPlaneState(BasePlaneState):
    # posture
    alpha: jax.typing.ArrayLike = 0.0
    beta: jax.typing.ArrayLike = 0.0
    # angular velocity
    P: jax.typing.ArrayLike = 0.0
    Q: jax.typing.ArrayLike = 0.0
    R: jax.typing.ArrayLike = 0.0
    # control state
    T: jax.typing.ArrayLike = 0.0
    el: jax.typing.ArrayLike = 0.0
    ail: jax.typing.ArrayLike = 0.0
    rud: jax.typing.ArrayLike = 0.0
    lef: jax.typing.ArrayLike = 0.0     # leading edge flap angle (deg)
    sb: jax.typing.ArrayLike = 0.0      # speed brake angle (rad)
    # acceleration
    ax: jax.typing.ArrayLike = 0.0
    ay: jax.typing.ArrayLike = 0.0
    az: jax.typing.ArrayLike = 0.0
    # perturbation parameters (default = identity / no perturbation)
    aero_scale: jax.typing.ArrayLike = 1.0
    mass_scale: jax.typing.ArrayLike = 1.0
    inertia_scale: jax.typing.ArrayLike = 1.0
    wind_body_u: jax.typing.ArrayLike = 0.0
    wind_body_v: jax.typing.ArrayLike = 0.0
    wind_body_w: jax.typing.ArrayLike = 0.0

    @classmethod
    def create(cls, state: jax.Array):
        return cls(
            north=state[0],
            east=state[1],
            altitude=state[2],
            roll=state[3],
            pitch=state[4],
            yaw=state[5],
            vel_x=state[6],
            vel_y=state[7],
            vel_z=state[8],
            vt=state[9],
            q0=state[10],
            q1=state[11],
            q2=state[12],
            q3=state[13],
            alpha=state[14],
            beta=state[15],
            P=state[16],
            Q=state[17],
            R=state[18],
            T=state[19],
            el=state[20],
            ail=state[21],
            rud=state[22],
            ax=state[23],
            ay=state[24],
            az=state[25],
        )


@struct.dataclass
class FighterPlaneControlState(BaseControlState):
    @classmethod
    def create(cls, action: jax.Array):
        return cls(
            throttle=action[0],
            elevator=action[1],
            aileron=action[2],
            rudder=action[3],
            speed_brake=action[4] if action.shape[0] > 4 else 0.0,
        )


def atmos(alt, vt):
    # 根据高度和速度计算动压、马赫数
    rho0 = 2.377e-3
    tfac = 1 - .703e-5 * (alt)
    temp = 519.0 * tfac
    temp = (alt >= 35000.0) * 390 + (alt < 35000.0) * temp
    rho = rho0 * jnp.pow(tfac, 4.14)
    mach = (vt) / jnp.sqrt(1.4 * 1716.3 * temp)
    qbar = .5 * rho * jnp.pow(vt, 2)
    ps = 1715.0 * rho * temp

    ps = (ps == 0) * 1715 + (ps != 0) * ps

    return (mach, qbar, ps)


def accels(roll, pitch, alpha, beta, vt, alpha_dot, beta_dot, vt_dot, P, Q, R):
    # 根据飞行状态结算三轴过载
    grav = 32.174
    sina = jnp.sin(alpha)
    cosa = jnp.cos(alpha)
    sinb = jnp.sin(beta)
    cosb = jnp.cos(beta)
    vel_u = vt * cosb * cosa
    vel_v = vt * sinb
    vel_w = vt * cosb * sina
    u_dot = cosb * cosa * vt_dot - vt * sinb * cosa * beta_dot - vt * cosb * sina * alpha_dot
    v_dot = sinb * vt_dot + vt * cosb * beta_dot
    w_dot = cosb * sina * vt_dot - vt * sinb * sina * beta_dot + vt * cosb * cosa * alpha_dot
    nx_cg = -1.0 / grav * (u_dot + Q * vel_w - R * vel_v) - jnp.sin(pitch)
    ny_cg = -1.0 / grav * (v_dot + R * vel_u - P * vel_w) + jnp.cos(pitch) * jnp.sin(roll)
    nz_cg = -1.0 / grav * (w_dot + P * vel_v - Q * vel_u) + jnp.cos(pitch) * jnp.cos(roll)
    return (nx_cg, ny_cg, nz_cg)

def quaternion_to_rpy(q0, q1, q2, q3):
    """
    将四元数(q0,q1,q2,q3)转换为欧拉角(roll, pitch, yaw).
    假定:
       - q0为实部, (q1,q2,q3)为空间部
       - 机体系 'body' -> 'inertial' 的Z-Y-X顺序
       - roll绕 x, pitch绕 y, yaw绕 z
    """
    '''
        Calculate attitude angle using quaternion q_{NED}^{Body}, return Eular angle in unit degree.
        注意：这里传入的四元数是 q_{NED}^{Body}，即从机体系到NED系的四元数。
        roll    Range [-180,180)
        pitch   Range [-90,90]
        yaw     Range [-180,180)
    '''
    sinr_cosp = 2.0 * (q0*q1 + q2*q3)
    cosr_cosp = 1.0 - 2.0 * (q1*q1 + q2*q2)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q0*q2 - q3*q1)
    pitch = jnp.where(jnp.abs(sinp) >= 1.0,
                      jnp.sign(sinp) * (jnp.pi / 2.0),
                      jnp.arcsin(sinp))

    siny_cosp = 2.0 * (q0*q3 + q1*q2)
    cosy_cosp = 1.0 - 2.0 * (q2*q2 + q3*q3)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw) 

def nlplant(xu, aero_scale=1.0, mass_scale=1.0, inertia_scale=1.0,
             wind_u=0.0, wind_v=0.0, wind_w=0.0):
    """
    model state(dim 16):
        0. north                 (unit: ft)
        1. east                  (unit: ft)
        2. altitude              (unit: ft)
        3. phi                   (unit: rad)
        4. theta                 (unit: rad)
        5. psi                   (unit: rad)
        6. vt                    (unit: ft/s)
        7. alpha                 (unit: rad)
        8. beta                  (unit: rad)
        9. P                     (unit: rad/s)
        10. Q                    (unit: rad/s)
        11. R                    (unit: rad/s)
        12. q0
        13. q1
        14. q2
        15. q3

    model control(dim 5)
        0. ego_T                  (unit: lbf)
        1. ego_el                 (unit: deg)
        2. ego_ail                (unit: deg)
        3. ego_rud                (unit: deg)
        4. ego_lef                (unit: deg)

    perturbation parameters (default = no perturbation):
        aero_scale:   multiplies all aerodynamic coefficients
        mass_scale:   multiplies aircraft mass
        inertia_scale: multiplies moments of inertia (Jx, Jy, Jz, Jxz)
        wind_u,v,w:   body-frame wind velocity (ft/s), subtracted from airspeed
    """
    xdot = jnp.zeros_like(xu)
    g = 32.17
    m = 636.94 * mass_scale
    B = 30.0
    S = 300.0
    cbar = 11.32
    xcgr = 0.35
    xcg = 0.30
    Heng = 0.0
    pi = jnp.pi

    Jy = 55814.0 * inertia_scale
    Jxz = 982.0 * inertia_scale
    Jz = 63100.0 * inertia_scale
    Jx = 9496.0 * inertia_scale

    r2d = 180.0 / pi

    # States
    alt = xu[2]

    vt = xu[6]
    alpha = xu[7] * r2d # deg
    beta = xu[8] * r2d  # deg
    P = xu[9]
    Q = xu[10]
    R = xu[11]

    # 用弧度计算三角函数
    sa = jnp.sin(xu[7])
    ca = jnp.cos(xu[7])
    sb = jnp.sin(xu[8])
    cb = jnp.cos(xu[8])

    vt = (vt <= 0.01) * 0.01 + (vt > 0.01) * vt

    ######################################################################
    # dynamics里面存的四元数是q_{Body}^{NED}，即从NED系到机体系的四元数（NED to Body）,而在转成欧拉角的时候需要转换为q_{NED}^{Body}，即从机体系到NED系的四元数（Body to NED），所以需要将q_{Body}^{NED}转换为q_{NED}^{Body}，即new_q0, new_q1, new_q2, new_q3转换为-new_q1, -new_q2, -new_q3, new_q0
    q0 = xu[12]
    q1 = xu[13]
    q2 = xu[14]
    q3 = xu[15]
    
    q0sq = q0**2
    q1sq = q1**2
    q2sq = q2**2
    q3sq = q3**2
    q0q1 = q0*q1
    q0q2 = q0*q2
    q0q3 = q0*q3
    q1q2 = q1*q2
    q1q3 = q1*q3
    q2q3 = q2*q3

    ######################################################################

    # Control inputs
    T = xu[16]
    el = xu[17]
    ail = xu[18]
    rud = xu[19]
    lef = xu[20]
    sb = xu[21] if xu.shape[0] > 21 else 0.0  # speed brake angle (rad)

    dail = ail / 21.5
    drud = rud / 30.0
    dlef = (1 - lef / 25.0)

    # Dynamics
    # Navigation Equations (ground-relative velocity)

    U = vt * ca * cb
    V = vt * sb
    W = vt * sa * cb

    xdot = xdot.at[0].set((q0sq+q1sq-q2sq-q3sq) * U + 2*(q1q2+q0q3)         * V + 2*(q1q3-q0q2)         * W) # VN
    xdot = xdot.at[1].set(2*(q1q2-q0q3)         * U + (q0sq-q1sq+q2sq-q3sq) * V + 2*(q2q3+q0q1)         * W) # VE
    xdot = xdot.at[2].set(-(2*(q1q3+q0q2)         * U + 2*(q2q3-q0q1)         * V + (q0sq-q1sq-q2sq+q3sq) * W)) # VU

    xdot = xdot.at[3].set(0.0) # roll
    xdot = xdot.at[4].set(0.0) # pitch
    xdot = xdot.at[5].set(0.0) # yaw

    # Air-relative velocity and effective aero angles
    # Use state alpha/beta as base; apply wind corrections for perturbation studies.
    # This avoids recomputing alpha/beta from velocity (which can produce
    # numerical differences even when wind=0 due to JAX JIT behaviour).
    U_air = U - wind_u
    V_air = V - wind_v
    W_air = W - wind_w
    vt_air = jnp.sqrt(U_air * U_air + V_air * V_air + W_air * W_air)
    vt_air = jnp.where(vt_air <= 0.01, 0.01, vt_air)
    # Wind-induced angle corrections (in degrees)
    alpha_wind_correction = (jnp.arctan2(W_air, U_air) - jnp.arctan2(W, U)) * r2d
    beta_wind_correction = (jnp.arcsin(jnp.clip(V_air / vt_air, -1.0, 1.0))
                           - jnp.arcsin(jnp.clip(V / vt, -1.0, 1.0))) * r2d
    alpha_eff = alpha + alpha_wind_correction
    beta_eff = beta + beta_wind_correction

    # Atmospheric effects (use air-relative speed)
    temp = atmos(alt, vt_air)
    mach = temp[0]
    qbar = temp[1]
    ps = temp[2]

    # Aero coefficients (state alpha/beta + wind correction)
    Cx = hifi_F16._Cx((el, beta_eff, alpha_eff))
    Cz = hifi_F16._Cz((el, beta_eff, alpha_eff))
    Cm = hifi_F16._Cm((el, beta_eff, alpha_eff))
    Cy = hifi_F16._Cy((beta_eff, alpha_eff))
    Cn = hifi_F16._Cn((el, beta_eff, alpha_eff))
    Cl = hifi_F16._Cl((el, beta_eff, alpha_eff))

    Cxq = hifi_F16._CXq(alpha_eff)
    Cyr = hifi_F16._CYr(alpha_eff)
    Cyp = hifi_F16._CYp(alpha_eff)
    Czq = hifi_F16._CZq(alpha_eff)
    Clr = hifi_F16._CLr(alpha_eff)
    Clp = hifi_F16._CLp(alpha_eff)
    Cmq = hifi_F16._CMq(alpha_eff)
    Cnr = hifi_F16._CNr(alpha_eff)
    Cnp = hifi_F16._CNp(alpha_eff)

    # ----- VERIFIED BUGFIX (2026-05-08) -----
    # The original code passed (alpha, beta) to bilinear LUTs that internally
    # call bilinear_interp(BETA1, ALPHA2, ...) — i.e. the function expects
    # (beta, alpha).  And it computed zero-elevator references via
    # _Cx((alpha, beta, 0)) which actually passes alpha into the elevator
    # slot and 0 into the alpha slot of the trilinear LUT
    # trilinear_interp(DH1, BETA1, ALPHA1, ...).
    #
    # This was verified against:
    #   1. The internal interp signatures in aero_data.py
    #   2. NASA TP-1538 Fortran reference (f16_deq.f) Cx table values:
    #      (DH, BETA, ALPHA) reshape gives RMSE=0.005-0.075 vs NASA — correct
    #      (ALPHA, BETA, DH) reshape gives RMSE=0.075-0.227 — wrong
    #   3. Physical consistency: corrected delta_Cx_lef has the expected
    #      sign reversal across stall (positive at small alpha, negative
    #      at large alpha), matching the documented LEF behaviour.
    #      The buggy version gives uniformly positive delta_Cx_lef
    #      monotonically growing with alpha — non-physical.
    #
    # Note: aero_data.py reshape order (DH, BETA, ALPHA) is CORRECT — do
    # not change it.  Main coefficient calls _Cx((el, beta, alpha)) above
    # are CORRECT — do not change them.  Only the LEF/a20/r30 differential
    # terms below were buggy.

    delta_Cx_lef = hifi_F16._Cx_lef((beta_eff, alpha_eff)) - hifi_F16._Cx((0.0, beta_eff, alpha_eff))
    delta_Cz_lef = hifi_F16._Cz_lef((beta_eff, alpha_eff)) - hifi_F16._Cz((0.0, beta_eff, alpha_eff))
    delta_Cm_lef = hifi_F16._Cm_lef((beta_eff, alpha_eff)) - hifi_F16._Cm((0.0, beta_eff, alpha_eff))
    delta_Cy_lef = hifi_F16._Cy_lef((beta_eff, alpha_eff)) - hifi_F16._Cy((beta_eff, alpha_eff))
    delta_Cn_lef = hifi_F16._Cn_lef((beta_eff, alpha_eff)) - hifi_F16._Cn((0.0, beta_eff, alpha_eff))
    delta_Cl_lef = hifi_F16._Cl_lef((beta_eff, alpha_eff)) - hifi_F16._Cl((0.0, beta_eff, alpha_eff))

    delta_Cxq_lef = hifi_F16._delta_CXq_lef(alpha_eff)
    delta_Cyr_lef = hifi_F16._delta_CYr_lef(alpha_eff)
    delta_Cyp_lef = hifi_F16._delta_CYp_lef(alpha_eff)
    delta_Clr_lef = hifi_F16._delta_CLr_lef(alpha_eff)
    delta_Clp_lef = hifi_F16._delta_CLp_lef(alpha_eff)
    delta_Cmq_lef = hifi_F16._delta_CMq_lef(alpha_eff)
    delta_Cnr_lef = hifi_F16._delta_CNr_lef(alpha_eff)
    delta_Cnp_lef = hifi_F16._delta_CNp_lef(alpha_eff)

    delta_Cy_r30 = hifi_F16._Cy_r30((beta_eff, alpha_eff)) - hifi_F16._Cy((beta_eff, alpha_eff))
    delta_Cn_r30 = hifi_F16._Cn_r30((beta_eff, alpha_eff)) - hifi_F16._Cn((0.0, beta_eff, alpha_eff))
    delta_Cl_r30 = hifi_F16._Cl_r30((beta_eff, alpha_eff)) - hifi_F16._Cl((0.0, beta_eff, alpha_eff))

    delta_Cy_a20 = hifi_F16._Cy_a20((beta_eff, alpha_eff)) - hifi_F16._Cy((beta_eff, alpha_eff))
    delta_Cy_a20_lef = hifi_F16._Cy_a20_lef((beta_eff, alpha_eff)) - hifi_F16._Cy_lef((beta_eff, alpha_eff)) -\
        (hifi_F16._Cy_a20((beta_eff, alpha_eff)) - hifi_F16._Cy((beta_eff, alpha_eff)))
    delta_Cn_a20 = hifi_F16._Cn_a20((beta_eff, alpha_eff)) - hifi_F16._Cn((0.0, beta_eff, alpha_eff))
    delta_Cn_a20_lef = hifi_F16._Cn_a20_lef((beta_eff, alpha_eff)) - hifi_F16._Cn_lef((beta_eff, alpha_eff)) -\
        (hifi_F16._Cn_a20((beta_eff, alpha_eff)) - hifi_F16._Cn((0.0, beta_eff, alpha_eff)))
    delta_Cl_a20 = hifi_F16._Cl_a20((beta_eff, alpha_eff)) - hifi_F16._Cl((0.0, beta_eff, alpha_eff))
    delta_Cl_a20_lef = hifi_F16._Cl_a20_lef((beta_eff, alpha_eff)) - hifi_F16._Cl_lef((beta_eff, alpha_eff)) -\
        (hifi_F16._Cl_a20((beta_eff, alpha_eff)) - hifi_F16._Cl((0.0, beta_eff, alpha_eff)))

    delta_Cnbeta = hifi_F16._delta_CNbeta(alpha_eff)
    delta_Clbeta = hifi_F16._delta_CLbeta(alpha_eff)
    delta_Cm = hifi_F16._delta_Cm(alpha_eff)
    eta_el = hifi_F16._eta_el(el)
    delta_Cm_ds = 0

    dXdQ = (cbar / (2 * vt_air + 1e-6)) * (Cxq + delta_Cxq_lef * dlef)
    Cx_tot = Cx + delta_Cx_lef * dlef + dXdQ * Q - hifi_F16._CDsb(alpha_eff) * sb
    dZdQ = (cbar / (2 * vt_air + 1e-6)) * (Czq + delta_Cz_lef * dlef)
    Cz_tot = Cz + delta_Cz_lef * dlef + dZdQ * Q - hifi_F16._CLsb(alpha_eff) * sb
    dMdQ = (cbar / (2 * vt_air + 1e-6)) * (Cmq + delta_Cmq_lef * dlef)
    Cm_tot = Cm * eta_el + Cz_tot * (xcgr - xcg) + delta_Cm_lef * dlef + dMdQ * Q + delta_Cm + delta_Cm_ds + hifi_F16._Cmsb(alpha_eff) * sb
    dYdail = delta_Cy_a20 + delta_Cy_a20_lef * dlef
    dYdR = (B / (2 * vt_air + 1e-6)) * (Cyr + delta_Cyr_lef * dlef)
    dYdP = (B / (2 * vt_air + 1e-6)) * (Cyp + delta_Cyp_lef * dlef)

    Cy_tot = Cy + delta_Cy_lef * dlef + dYdail * dail + delta_Cy_r30 * drud + dYdR * R + dYdP * P
    dNdail = delta_Cn_a20 + delta_Cn_a20_lef * dlef
    dNdR = (B / (2 * vt_air + 1e-6)) * (Cnr + delta_Cnr_lef * dlef)
    dNdP = (B / (2 * vt_air + 1e-6)) * (Cnp + delta_Cnp_lef * dlef)
    Cn_tot = Cn + delta_Cn_lef * dlef - Cy_tot * (xcgr - xcg) * (cbar / B) + dNdail * dail + delta_Cn_r30 * drud + dNdR * R + dNdP * P + delta_Cnbeta * beta_eff
    dLdail = delta_Cl_a20 + delta_Cl_a20_lef * dlef
    dLdR = (B / (2 * vt_air + 1e-6)) * (Clr + delta_Clr_lef * dlef)
    dLdP = (B / (2 * vt_air + 1e-6)) * (Clp + delta_Clp_lef * dlef)
    Cl_tot = Cl + delta_Cl_lef * dlef + dLdail * dail + delta_Cl_r30 * drud + dLdR * R + dLdP * P + delta_Clbeta * beta_eff

    ######################################################################
    Udot = R * V - Q * W + g * 2*(q1q3+q0q2) + qbar * S * Cx_tot * aero_scale / m + T / m
    Vdot = P * W - R * U + g * 2*(q2q3-q0q1) + qbar * S * Cy_tot * aero_scale / m
    Wdot = Q * U - P * V + g * (q0sq-q1sq-q2sq+q3sq) + qbar * S * Cz_tot * aero_scale / m
    ######################################################################

    xdot = xdot.at[6].set((U * Udot + V * Vdot + W * Wdot) / (vt + 1e-6)) # Vt
    xdot = xdot.at[7].set((U * Wdot - W * Udot) / (U * U + W * W + 1e-6)) # alpha
    xdot = xdot.at[8].set((Vdot * vt - V * xdot[6]) / (vt * vt * cb + 1e-6)) # beta

    L_tot = Cl_tot * qbar * S * B * aero_scale
    M_tot = Cm_tot * qbar * S * cbar * aero_scale
    N_tot = Cn_tot * qbar * S * B * aero_scale
    denom = Jx * Jz - Jxz * Jxz + 1e-6

    xdot = xdot.at[9].set((Jz * L_tot + Jxz * N_tot - (Jz * (Jz - Jy) + Jxz * Jxz) * Q * R + Jxz * (Jx - Jy + Jz) * P * Q + Jxz * Q * Heng) / denom) # P
    xdot = xdot.at[10].set((M_tot + (Jz - Jx) * P * R - Jxz * (P * P - R * R) - R * Heng) / Jy) # Q
    xdot = xdot.at[11].set((Jx * N_tot + Jxz * L_tot + (Jx * (Jx - Jy) + Jxz * Jxz) * P * Q - Jxz * (Jx - Jy + Jz) * Q * R + Jx * Q * Heng) / denom) # R

    # Quaternion Kinematics (NED to Body)
    xdot = xdot.at[12].set(0.5*(      P*q1+Q*q2+R*q3)) # q0
    xdot = xdot.at[13].set(0.5*(-P*q0     +R*q2-Q*q3)) # q1
    xdot = xdot.at[14].set(0.5*(-Q*q0-R*q1     +P*q3)) # q2
    xdot = xdot.at[15].set(0.5*(-R*q0+Q*q1-P*q2     )) # q3

    return xdot

def update(state: FighterPlaneState, action: FighterPlaneControlState, dt: float) -> FighterPlaneState:
    # 1) 构造 x 向量 (长度16)
    x = jnp.hstack((
        state.north / 0.3048,    # 0
        state.east / 0.3048,     # 1
        state.altitude / 0.3048, # 2
        state.roll,              # 3
        state.pitch,             # 4
        state.yaw,               # 5
        state.vt / 0.3048,       # 6
        state.alpha,             # 7
        state.beta,              # 8
        state.P,                 # 9
        state.Q,                 # 10
        state.R,                 # 11
        state.q0,                # 12
        state.q1,                # 13
        state.q2,                # 14
        state.q3                 # 15
    ))

    T = 0.9 * state.T + 0.1 * action.throttle * 0.225 * 76300 / 0.3048
    el  = 0.9 * state.el  + 0.1 * action.elevator * 45
    ail = 0.9 * state.ail + 0.1 * action.aileron  * 45
    rud = 0.9 * state.rud + 0.1 * action.rudder   * 45

    # ---- LEF auto-scheduling (JSBSim-style, based on alpha and Mach) ----
    alpha_deg = state.alpha * 180.0 / jnp.pi
    vt_fts = state.vt / 0.3048
    alt_ft = state.altitude / 0.3048
    rho0 = 2.377e-3
    tfac = 1 - .703e-5 * alt_ft
    temp_R = 519.0 * tfac
    temp_R = jnp.where(alt_ft >= 35000.0, 390.0, temp_R)
    mach = vt_fts / jnp.sqrt(1.4 * 1716.3 * temp_R)

    lef_cmd = jnp.where(mach > 0.9, -2.0,                    # supersonic: retract
               jnp.where(alpha_deg > 15.0, 25.0,             # high alpha: full deploy
               jnp.where(alpha_deg > 5.0, 15.0,              # moderate alpha: partial
               0.0)))                                         # low alpha: retracted
    lef_coef = jnp.exp(-dt / 0.5)                             # τ=0.5s → 1.5s for 0°→25°
    lef = lef_coef * state.lef + (1.0 - lef_coef) * lef_cmd

    # ---- speed brake (JSBSim-style, 0–60° range) ----
    sb_cmd = action.speed_brake * jnp.deg2rad(60.0)
    sb = 0.9 * state.sb + 0.1 * sb_cmd                       # τ≈0.19s, ~1s full deploy

    u = jnp.hstack((T, el, ail, rud, lef, sb))
    xu = jnp.hstack((x, u))

    xdot = nlplant(
        xu,
        aero_scale=state.aero_scale,
        mass_scale=state.mass_scale,
        inertia_scale=state.inertia_scale,
        wind_u=state.wind_body_u / 0.3048,    # convert m/s → ft/s
        wind_v=state.wind_body_v / 0.3048,
        wind_w=state.wind_body_w / 0.3048,
    )

    nx_cg, ny_cg, nz_cg = accels(
        xu[3], xu[4], xu[7], xu[8], xu[6], 
        xdot[7], xdot[8], xdot[6],
        xu[9], xu[10], xu[11]
    )

    new_x = x + xdot[:16] * dt

    # dynamics里面存的四元数是q_{Body}^{NED}，即从NED系到机体系的四元数（NED to Body）,而在转成欧拉角的时候需要转换为q_{NED}^{Body}，即从机体系到NED系的四元数（Body to NED），所以需要将q_{Body}^{NED}转换为q_{NED}^{Body}，即new_q0, new_q1, new_q2, new_q3转换为-new_q1, -new_q2, -new_q3, new_q0
    new_q0 = new_x[12]
    new_q1 = new_x[13]
    new_q2 = new_x[14]
    new_q3 = new_x[15]
    norm_q = jnp.sqrt(new_q0**2 + new_q1**2 + new_q2**2 + new_q3**2) + 1e-6
    new_q0 /= norm_q
    new_q1 /= norm_q
    new_q2 /= norm_q
    new_q3 /= norm_q

    ####################################################################################
    # --- Keep quaternion on same hemisphere (continuity fix) ---
    dot  = state.q0 * new_q0 + state.q1 * new_q1 + state.q2 * new_q2 + state.q3 * new_q3
    sign = jnp.where(dot < 0.0, -1.0, 1.0)
    new_q0 = new_q0 * sign
    new_q1 = new_q1 * sign
    new_q2 = new_q2 * sign
    new_q3 = new_q3 * sign
    # ------------------------------------------------------------
    ####################################################################################

    roll, pitch, yaw = quaternion_to_rpy(new_q0, -new_q1, -new_q2, -new_q3)

    new_state = state.replace(
        north=jnp.nan_to_num(new_x[0] * 0.3048, nan=0.0),
        east=jnp.nan_to_num(new_x[1] * 0.3048, nan=0.0),
        altitude=jnp.nan_to_num(new_x[2] * 0.3048, nan=0.0),
        roll=jnp.nan_to_num(roll, nan=0.0),
        pitch=jnp.nan_to_num(pitch, nan=0.0),
        yaw=jnp.nan_to_num(yaw, nan=0.0),
        vel_x=jnp.nan_to_num(xdot[0] * 0.3048, nan=0.0),
        vel_y=jnp.nan_to_num(xdot[1] * 0.3048, nan=0.0),
        vel_z=jnp.nan_to_num(xdot[2] * 0.3048, nan=0.0),
        vt=jnp.nan_to_num(new_x[6] * 0.3048, nan=0.0),
        alpha=jnp.nan_to_num(new_x[7], nan=0.0),
        beta=jnp.nan_to_num(new_x[8], nan=0.0),
        P=jnp.nan_to_num(new_x[9], nan=0.0),
        Q=jnp.nan_to_num(new_x[10], nan=0.0),
        R=jnp.nan_to_num(new_x[11], nan=0.0),
        q0=jnp.nan_to_num(new_q0, nan=1.0),
        q1=jnp.nan_to_num(new_q1, nan=0.0),
        q2=jnp.nan_to_num(new_q2, nan=0.0),
        q3=jnp.nan_to_num(new_q3, nan=0.0),
        T=jnp.nan_to_num(T, nan=0.0),
        el=jnp.nan_to_num(el, nan=0.0),
        ail=jnp.nan_to_num(ail, nan=0.0),
        rud=jnp.nan_to_num(rud, nan=0.0),
        lef=jnp.nan_to_num(lef, nan=0.0),
        sb=jnp.nan_to_num(sb, nan=0.0),
        ax=jnp.nan_to_num(nx_cg, nan=0.0),
        ay=jnp.nan_to_num(ny_cg, nan=0.0),
        az=jnp.nan_to_num(nz_cg, nan=0.0),
        # preserve perturbation parameters
        aero_scale=state.aero_scale,
        mass_scale=state.mass_scale,
        inertia_scale=state.inertia_scale,
        wind_body_u=state.wind_body_u,
        wind_body_v=state.wind_body_v,
        wind_body_w=state.wind_body_w,
    )
    mask = state.is_alive | state.is_locked
    state = jax.lax.cond(mask, lambda: new_state, lambda: state)
    return state

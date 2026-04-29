"""Parameterized trajectories with analytic kinematics and simple limit enforcement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

Vec3 = Tuple[float, float, float]
# Quaternion components in xyzw order (matches ``set_camera_pose`` in blender.py).
QuatXYZW = Tuple[float, float, float, float]
State = Tuple[Vec3, QuatXYZW, Vec3, Vec3]


@dataclass
class TrajectoryConfig:
    """Shared parameters and physics limits (SI-style units; tune as needed)."""

    z: float = 3.0
    figure8_A: float = 2.0
    omega: float = 0.35
    circle_R: float = 2.0
    hover_yaw_rate: float = 0.25
    hover_xy: Tuple[float, float] = (0.0, 0.0)
    max_linear_accel: float = 8.0
    max_angular_rate: float = 1.25
    max_roll_pitch_deg: float = 45.0


def _heading_quaternion_xy(vx: float, vy: float, eps: float = 1e-9) -> QuatXYZW:
    """Yaw-only orientation so camera +X aligns with horizontal velocity (level flight)."""
    speed_sq = vx * vx + vy * vy
    if speed_sq < eps * eps:
        return (0.0, 0.0, 0.0, 1.0)
    psi = math.atan2(vy, vx)
    half = 0.5 * psi
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _check_roll_pitch(quat_xyzw: QuatXYZW, limit_deg: float) -> None:
    """Sanity check: |roll|, |pitch| derived from quaternion should stay within limit."""
    qx, qy, qz, qw = quat_xyzw
    # Tait–Bryan yaw–pitch–roll (XYZ intrinsic) from quaternion.
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    lim = math.radians(limit_deg)
    # yaw unconstrained
    if abs(roll) > lim + 1e-6 or abs(pitch) > lim + 1e-6:
        raise RuntimeError(
            f"roll/pitch exceed {limit_deg}° (roll={math.degrees(roll):.3f}, pitch={math.degrees(pitch):.3f})"
        )


def _planar_speed_and_heading_rate(vx: float, vy: float, ax: float, ay: float, eps: float = 1e-9) -> Tuple[float, float]:
    v2 = vx * vx + vy * vy
    if v2 < eps * eps:
        return 0.0, 0.0
    speed = math.sqrt(v2)
    cross = vx * ay - vy * ax
    psi_dot = cross / v2
    return speed, psi_dot


def _circle_omega_cap(cfg: TrajectoryConfig) -> float:
    """|a| = R w^2  =>  w <= sqrt(a_max / R)."""
    if cfg.circle_R <= 1e-9:
        return cfg.omega
    w_cap = math.sqrt(max(cfg.max_linear_accel / cfg.circle_R, 0.0))
    return min(cfg.omega, w_cap)


def _figure8_scale_omega(cfg: TrajectoryConfig, samples: int = 256) -> float:
    """Reduce omega until sampled max |a| <= cfg.max_linear_accel."""
    A = cfg.figure8_A
    omega_try = cfg.omega
    for _ in range(32):
        amax = 0.0
        for k in range(samples):
            theta = 2.0 * math.pi * k / samples
            ax = -A * omega_try * omega_try * math.sin(theta)
            ay = -A * omega_try * omega_try * 2.0 * math.sin(2.0 * theta)
            amax = max(amax, math.hypot(ax, ay))
        if amax <= cfg.max_linear_accel + 1e-6:
            return omega_try
        scale = math.sqrt(cfg.max_linear_accel / max(amax, 1e-12))
        omega_try *= scale
    return omega_try


def circle(t: float, cfg: TrajectoryConfig | None = None) -> State:
    cfg = cfg or TrajectoryConfig()
    w = _circle_omega_cap(cfg)
    wt = w * t
    R = cfg.circle_R
    x = R * math.cos(wt)
    y = R * math.sin(wt)
    z = cfg.z
    vx = -R * w * math.sin(wt)
    vy = R * w * math.cos(wt)
    vz = 0.0
    ax = -R * w * w * math.cos(wt)
    ay = -R * w * w * math.sin(wt)
    az = 0.0
    quat = _heading_quaternion_xy(vx, vy)
    _, psi_dot = _planar_speed_and_heading_rate(vx, vy, ax, ay)
    if abs(psi_dot) > cfg.max_angular_rate + 1e-6:
        raise RuntimeError(f"heading rate {psi_dot} exceeds max_angular_rate {cfg.max_angular_rate}")
    _check_roll_pitch(quat, cfg.max_roll_pitch_deg)
    return ((x, y, z), quat, (vx, vy, vz), (ax, ay, az))


def figure8(t: float, cfg: TrajectoryConfig | None = None) -> State:
    cfg = cfg or TrajectoryConfig()
    A = cfg.figure8_A
    z = cfg.z
    w = _figure8_scale_omega(cfg)
    theta = w * t
    st, ct = math.sin(theta), math.cos(theta)
    x = A * st
    y = A * st * ct
    vx = A * w * ct
    vy = A * w * math.cos(2.0 * theta)
    vz = 0.0
    ax = -A * w * w * st
    ay = -A * w * w * 2.0 * math.sin(2.0 * theta)
    az = 0.0
    quat = _heading_quaternion_xy(vx, vy)
    _, psi_dot = _planar_speed_and_heading_rate(vx, vy, ax, ay)
    yaw_rate = abs(psi_dot)
    if yaw_rate > cfg.max_angular_rate + 1e-6:
        raise RuntimeError(
            f"heading angular rate {yaw_rate} exceeds max_angular_rate {cfg.max_angular_rate}"
        )
    _check_roll_pitch(quat, cfg.max_roll_pitch_deg)
    return ((x, y, z), quat, (vx, vy, vz), (ax, ay, az))


def hover_with_yaw(t: float, cfg: TrajectoryConfig | None = None) -> State:
    cfg = cfg or TrajectoryConfig()
    hx, hy = cfg.hover_xy
    z = cfg.z
    w_yaw = min(abs(cfg.hover_yaw_rate), cfg.max_angular_rate)
    if cfg.hover_yaw_rate < 0:
        w_yaw = -w_yaw
    psi = w_yaw * t
    half = 0.5 * psi
    quat = (0.0, 0.0, math.sin(half), math.cos(half))
    _check_roll_pitch(quat, cfg.max_roll_pitch_deg)
    pos = (hx, hy, z)
    vel = (0.0, 0.0, 0.0)
    acc = (0.0, 0.0, 0.0)
    lin_mag = math.sqrt(acc[0] ** 2 + acc[1] ** 2 + acc[2] ** 2)
    if lin_mag > cfg.max_linear_accel + 1e-9:
        raise RuntimeError("linear acceleration exceeds limit")
    return (pos, quat, vel, acc)


def velocity_finite_difference_consistency(
    traj: Callable[[float], State],
    t: float,
    eps: float = 1e-5,
    rtol: float = 5e-3,
    atol: float = 1e-4,
) -> bool:
    """True if central difference of position matches analytic velocity within tol."""
    p_m, _, v_m, _ = traj(t - eps)
    p_p, _, v_p, _ = traj(t + eps)
    fd = (
        (p_p[0] - p_m[0]) / (2.0 * eps),
        (p_p[1] - p_m[1]) / (2.0 * eps),
        (p_p[2] - p_m[2]) / (2.0 * eps),
    )
    v_mid = (
        0.5 * (v_m[0] + v_p[0]),
        0.5 * (v_m[1] + v_p[1]),
        0.5 * (v_m[2] + v_p[2]),
    )
    for i in range(3):
        if abs(fd[i] - v_mid[i]) > atol + rtol * max(abs(v_mid[i]), abs(fd[i]), 1.0):
            return False
    return True

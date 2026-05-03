"""
Trajectory implementations for Phase 2 synthetic data generation.

Trajectory concept (designed by Myrrh):
- Line & Square: Start with acceleration = 0, increase to peak at middle, decrease until both
  acceleration and velocity = 0 at end of segment, then turn (180° for line, 90° for square),
  then repeat pattern.
- Figure8, Circle: Forward velocity is constant; xyz change to follow the path.
- Moon: Like figure8/circle but on a 3D sphere; net velocity remains constant while xyz varies.

Implementation architecture (designed by Copilot):
- Abstract base class with precomputation and caching for 1000 Hz evaluation.
- Each trajectory subclass implements three parametric functions: acceleration, velocity, position.
- Base class provides get_state(t) for fast O(1) lookups and quaternion computation.
"""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

try:
    from .ensure_installed import ensure_installed
except ImportError:
    from ensure_installed import ensure_installed  # type: ignore

ensure_installed("numpy")
ensure_installed("scipy")

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation.

    Written by: Copilot
    """

    duration: float = 10.0      # Total trajectory duration in seconds
    max_velocity: float = 1.0   # Peak velocity (line/square) or constant speed (curves)


class Trajectory(ABC):
    """Abstract base trajectory class with precomputation and caching.

    Design (Myrrh): Trajectories are deterministic motion profiles.
    Implementation (Copilot): Pre-sample at 1000 Hz during init, cache as splines for fast O(1) evaluation.
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0):
        """Initialize trajectory and precompute all parametric functions.

        Args:
            duration: Total trajectory time in seconds
            max_velocity: Peak velocity (line/square) or constant speed (curves)

        Implementation written by: Copilot
        """
        self.duration = duration
        self.max_velocity = max_velocity
        self._precompute_functions()

    def _precompute_functions(self):
        """Sample trajectory at 1000 Hz and store as fast interpolants.

        Written by: Copilot
        """
        dt = 0.001  # 1000 Hz sample rate
        times = np.arange(0, self.duration, dt)

        accels = []
        velocities = []
        positions = []

        for t in times:
            accels.append(self.acceleration_func(t))
            velocities.append(self.velocity_func(t))
            positions.append(self.position_func(t))

        accels = np.array(accels)
        velocities = np.array(velocities)
        positions = np.array(positions)

        self.accel_interp = CubicSpline(times, accels, axis=0)
        self.vel_interp = CubicSpline(times, velocities, axis=0)
        self.pos_interp = CubicSpline(times, positions, axis=0)

    @abstractmethod
    def acceleration_func(self, t: float) -> np.ndarray:
        """Return acceleration (ax, ay, az) at time t."""
        pass

    @abstractmethod
    def velocity_func(self, t: float) -> np.ndarray:
        """Return velocity (vx, vy, vz) at time t."""
        pass

    @abstractmethod
    def position_func(self, t: float) -> np.ndarray:
        """Return position (x, y, z) at time t."""
        pass

    def get_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (position, quaternion, velocity, acceleration) at time t.

        Fast lookup using precomputed splines. Quaternion derived from velocity and acceleration.
        Moon altitude offset applied here so position_func stays a pure geometric curve.

        Written by: Copilot
        """
        t = np.clip(t, 0, self.duration - 0.001)

        p = self.pos_interp(t)
        v = self.vel_interp(t)
        a = self.accel_interp(t)

        # Moon.position_func returns pure sphere coords; altitude lives here, not in the curve.
        # All other trajectories already embed z=2.0 in position_func directly.
        if isinstance(self, Moon):
            p = p + np.array([0, 0, 2.0])

        q = self._velocity_accel_to_quaternion(v, a)
        return p, q, v, a

    def _velocity_accel_to_quaternion(self, velocity: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """Compute quaternion from velocity direction and acceleration.

        - Yaw: derived from velocity direction in x-y plane
        - Pitch/Roll: small tilt based on acceleration perpendicular components

        Written by: Copilot
        """
        v_norm = np.linalg.norm(velocity)
        if v_norm < 1e-9:
            return np.array([0, 0, 0, 1])  # [x, y, z, w] format

        yaw = np.arctan2(velocity[1], velocity[0])
        pitch = np.arcsin(np.clip(accel[2] / (9.81 + 1e-9), -0.3, 0.3)) * 0.1
        roll = np.arcsin(np.clip(-accel[1] / (9.81 + 1e-9), -0.3, 0.3)) * 0.1

        rotation = Rotation.from_euler("xyz", [roll, pitch, yaw])
        return rotation.as_quat()  # Returns [x, y, z, w]


class Line(Trajectory):
    """Two straight passes with 180° turnaround.

    Trajectory design (Myrrh):
    - Segment 1: Accelerate from rest, reach peak speed at middle, decelerate to rest
    - Turn: Rotate 180° over 1 second (stationary)
    - Segment 2: Repeat in reverse direction
    - End of dataset

    Implementation (Copilot):
    - Linear acceleration ramp: increases linearly first half, decreases linearly second half
    - Velocity integrates to quadratic
    - Position integrates to cubic
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0):
        """Initialize Line trajectory.

        Args:
            duration: Total trajectory time (default 10 seconds)
            max_velocity: Peak velocity during segments (default 1 m/s)

        Design written by: Myrrh
        Implementation written by: Copilot
        """
        self.seg_duration = (duration - 1.0) / 2   # Each segment ~4.5 sec if duration=10
        self.turn_start = self.seg_duration
        self.turn_end = self.seg_duration + 1.0
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Linear acceleration ramp up then down.

        Written by: Copilot (implementation of Myrrh's design)
        """
        if t < self.turn_start:
            a_mag = self._linear_ramp_accel(t, self.seg_duration)
            return np.array([a_mag, 0, 0])
        elif t < self.turn_end:
            return np.array([0, 0, 0])
        else:
            seg_t = t - self.turn_end
            a_mag = self._linear_ramp_accel(seg_t, self.seg_duration)
            return np.array([-a_mag, 0, 0])

    def velocity_func(self, t: float) -> np.ndarray:
        """Quadratic velocity profile (integrating linear acceleration).

        Written by: Copilot
        """
        if t < self.turn_start:
            v_mag = self._linear_ramp_velocity(t, self.seg_duration)
            return np.array([v_mag, 0, 0])
        elif t < self.turn_end:
            return np.array([0, 0, 0])
        else:
            seg_t = t - self.turn_end
            v_mag = self._linear_ramp_velocity(seg_t, self.seg_duration)
            return np.array([-v_mag, 0, 0])

    def position_func(self, t: float) -> np.ndarray:
        """Cubic position profile (integrating quadratic velocity).

        Written by: Copilot
        """
        z = 2.0
        if t < self.turn_start:
            x = self._linear_ramp_position(t, self.seg_duration)
            return np.array([x, 0, z])
        elif t < self.turn_end:
            x_max = self._linear_ramp_position(self.seg_duration, self.seg_duration)
            return np.array([x_max, 0, z])
        else:
            seg_t = t - self.turn_end
            delta_x = self._linear_ramp_position(seg_t, self.seg_duration)
            x_max = self._linear_ramp_position(self.seg_duration, self.seg_duration)
            return np.array([x_max - delta_x, 0, z])

    def _linear_ramp_accel(self, t: float, seg_dur: float) -> float:
        """Linear acceleration ramp: up first half, down second half.

        a(t) = a_peak * (t / half),             0 <= t < half
        a(t) = a_peak * (1 - (t-half) / half),  half <= t < seg_dur

        v_max = ∫₀^half a_peak*(t/half) dt
              = (a_peak/half) * [t²/2]₀^half
              = a_peak * half / 2
        => a_peak = 2*v_max / half

        Written by: Copilot
        """
        half = seg_dur / 2
        a_peak = 2 * self.max_velocity / half   # = 4*v_max/seg_dur
        if t < half:
            return a_peak * (t / half)
        else:
            return a_peak * (1 - (t - half) / half)

    def _linear_ramp_velocity(self, t: float, seg_dur: float) -> float:
        """Quadratic velocity from integrating linear acceleration.

        v(t) = ∫₀^t a_peak*(s/half) ds
             = (a_peak / half) * t²/2
             = (2*v_max/half²) * t²/2
             = v_max * (t/half)²

        Decel phase is symmetric: v(t) = v_max * (1 - ((t-half)/half)²)

        Written by: Copilot
        """
        half = seg_dur / 2
        if t < half:
            return self.max_velocity * (t / half) ** 2
        else:
            return self.max_velocity * (1 - ((t - half) / half) ** 2)

    def _linear_ramp_position(self, t: float, seg_dur: float) -> float:
        """Cubic position from integrating quadratic velocity.

        p(t) = ∫₀^t v_max*(s/half)² ds
             = v_max * t³ / (3*half²)
             = (v_max*half/3) * (t/half)³

        Total segment distance = 2 * (v_max*half/3)  [accel + decel by symmetry]

        Decel phase: p(t) = seg_dist - (v_max*half/3) * ((t-half)/half)³

        Written by: Copilot
        """
        half = seg_dur / 2
        seg_dist = 2 * self.max_velocity * half / 3
        if t < half:
            return (self.max_velocity * half / 3) * (t / half) ** 3
        else:
            return seg_dist - (self.max_velocity * half / 3) * ((t - half) / half) ** 3


class Square(Trajectory):
    """Four sides with 90° turns.

    Trajectory design (Myrrh):
    - Repeat 4 times:
      - Accelerate from rest, peak at middle, decelerate to rest
      - Turn 90° over 1 second (stationary)
    - End of dataset

    Implementation (Copilot):
    - Cycles through 4 segments (x, y, -x, -y directions)
    - Same linear accel/decel as Line, but 4 repetitions with 90° turns
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0):
        """Initialize Square trajectory.

        Design written by: Myrrh
        Implementation written by: Copilot
        """
        self.turn_duration = 1.0
        self.seg_duration = (duration - 4 * self.turn_duration) / 4
        self.cycle_duration = self.seg_duration + self.turn_duration
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Cycle through 4 segments with linear ramp each.

        Written by: Copilot
        """
        cycle_num = int(t / self.cycle_duration)
        t_in_cycle = t - cycle_num * self.cycle_duration

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = directions[cycle_num % 4]

        if t_in_cycle < self.seg_duration:
            a_mag = self._linear_ramp_accel(t_in_cycle, self.seg_duration)
            return np.array([a_mag * dx, a_mag * dy, 0])
        else:
            return np.array([0, 0, 0])

    def velocity_func(self, t: float) -> np.ndarray:
        """Cycle through 4 segments with quadratic velocity.

        Written by: Copilot
        """
        cycle_num = int(t / self.cycle_duration)
        t_in_cycle = t - cycle_num * self.cycle_duration

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = directions[cycle_num % 4]

        if t_in_cycle < self.seg_duration:
            v_mag = self._linear_ramp_velocity(t_in_cycle, self.seg_duration)
            return np.array([v_mag * dx, v_mag * dy, 0])
        else:
            return np.array([0, 0, 0])

    def position_func(self, t: float) -> np.ndarray:
        """Position along square perimeter, accumulating corner offsets.

        Each leg starts from the corner where the previous leg ended:
        - corner[n] = sum of direction[i] * seg_distance for i in 0..n-1
        - current pos = corner[cycle_num] + direction[cycle_num] * seg_dist_so_far

        Written by: Copilot
        """
        z = 2.0
        cycle_num = int(t / self.cycle_duration)
        t_in_cycle = t - cycle_num * self.cycle_duration

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        seg_distance = self._linear_ramp_position(self.seg_duration, self.seg_duration)

        # Accumulate corner position from all completed legs
        ox, oy = 0.0, 0.0
        for i in range(cycle_num):
            ddx, ddy = directions[i % 4]
            ox += ddx * seg_distance
            oy += ddy * seg_distance

        # Current progress within this leg
        if t_in_cycle < self.seg_duration:
            seg_dist = self._linear_ramp_position(t_in_cycle, self.seg_duration)
        else:
            seg_dist = seg_distance

        dx, dy = directions[cycle_num % 4]
        return np.array([ox + dx * seg_dist, oy + dy * seg_dist, z])

    def _linear_ramp_accel(self, t: float, seg_dur: float) -> float:
        """Linear acceleration ramp.

        a_peak = 2*v_max / half  (see Line._linear_ramp_accel for full derivation)

        Written by: Copilot
        """
        half = seg_dur / 2
        a_peak = 2 * self.max_velocity / half
        if t < half:
            return a_peak * (t / half)
        else:
            return a_peak * (1 - (t - half) / half)

    def _linear_ramp_velocity(self, t: float, seg_dur: float) -> float:
        """Quadratic velocity from linear acceleration.

        v(t) = v_max*(t/half)²  (see Line._linear_ramp_velocity for full derivation)

        Written by: Copilot
        """
        half = seg_dur / 2
        if t < half:
            return self.max_velocity * (t / half) ** 2
        else:
            return self.max_velocity * (1 - ((t - half) / half) ** 2)

    def _linear_ramp_position(self, t: float, seg_dur: float) -> float:
        """Cubic position from quadratic velocity.

        p(t) = (v_max*half/3)*(t/half)³  (see Line._linear_ramp_position for full derivation)

        Written by: Copilot
        """
        half = seg_dur / 2
        seg_dist = 2 * self.max_velocity * half / 3
        if t < half:
            return (self.max_velocity * half / 3) * (t / half) ** 3
        else:
            return seg_dist - (self.max_velocity * half / 3) * ((t - half) / half) ** 3


class Circle(Trajectory):
    """Circular path at constant forward speed.

    Trajectory design (Myrrh):
    - Forward velocity is constant throughout
    - XYZ change to follow circle path
    - Completes full circle over trajectory duration

    Implementation (Copilot):
    - Parametric circle: x = R*cos(θ), y = R*sin(θ),  θ(t) = 2π*t/T
    - Velocity tangent to circle with constant magnitude |v| = R*dθ/dt = v_max
    - Acceleration is centripetal: |a| = v_max²/R, directed toward center
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0, radius: float = None):
        """Initialize Circle trajectory.

        Args:
            duration: Total time to complete circle
            max_velocity: Constant forward speed
            radius: Circle radius in meters. If None, derived from v_max and duration:
                    R = v_max * T / (2π), which guarantees |v| = v_max exactly.
                    If provided, the actual speed will be R*2π/T, not v_max.

        |v| = R * dθ/dt = R * (2π/T) = v_max  =>  R = v_max*T/(2π)

        Design written by: Myrrh
        Implementation written by: Copilot
        """
        self.radius = radius if radius is not None else max_velocity * duration / (2 * np.pi)
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Centripetal acceleration pointing toward center.

        a(t) = -ω² * p(t),  ω = dθ/dt = 2π/T
        |a| = v_max² / R  (constant magnitude, direction rotates)

        Written by: Copilot
        """
        theta = (t / self.duration) * 2 * np.pi
        a_mag = (self.max_velocity ** 2) / self.radius
        return np.array([-a_mag * np.cos(theta), -a_mag * np.sin(theta), 0])

    def velocity_func(self, t: float) -> np.ndarray:
        """Tangent to circle with constant speed.

        v(t) = dp/dt = R*ω * (-sin θ, cos θ)
        |v| = R*ω = R*(2π/T) = v_max  (constant)

        Written by: Copilot
        """
        theta = (t / self.duration) * 2 * np.pi
        dtheta_dt = 2 * np.pi / self.duration
        return np.array([
            -self.radius * np.sin(theta) * dtheta_dt,
             self.radius * np.cos(theta) * dtheta_dt,
             0,
        ])

    def position_func(self, t: float) -> np.ndarray:
        """Parametric circle position.

        p(t) = (R*cos θ, R*sin θ, z),  θ = 2π*t/T

        Written by: Copilot
        """
        theta = (t / self.duration) * 2 * np.pi
        return np.array([self.radius * np.cos(theta), self.radius * np.sin(theta), 2.0])


class Figure8(Trajectory):
    """Figure-8 (lemniscate) path at constant forward speed.

    Trajectory design (Myrrh):
    - Forward velocity is constant throughout
    - XYZ change to follow figure-8 path
    - More complex than circle, loops twice

    Implementation (Copilot):
    - Parametric lemniscate: x = A*sin(u), y = A*sin(u)*cos(u),  u = 2π*t/T
    - Velocity tangent normalized then scaled to v_max (arc-length reparametrization)
    - Acceleration via central-difference on position_func (d²p/dt² ≈ Δ²p/Δt²)
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0, amplitude: float = 1.0):
        """Initialize Figure8 trajectory.

        Args:
            duration: Total time
            max_velocity: Constant forward speed
            amplitude: Scale of figure-8 pattern

        Design written by: Myrrh
        Implementation written by: Copilot
        """
        self.amplitude = amplitude
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Second derivative of position via central difference.

        a(t) ≈ (p(t+ε) - 2p(t) + p(t-ε)) / ε²

        Differencing position_func (not velocity_func) avoids spikes near the
        lemniscate crossover where the tangent normalization denominator → 0.

        Written by: Copilot
        """
        eps = 0.0001
        t_lo = max(t - eps, 0)
        t_hi = min(t + eps, self.duration)
        return (
            self.position_func(t_hi)
            - 2 * self.position_func(t)
            + self.position_func(t_lo)
        ) / (eps ** 2)

    def velocity_func(self, t: float) -> np.ndarray:
        """Tangent to lemniscate, scaled to constant speed.

        Raw tangent: dp/du = A*(cos u, cos 2u, 0)
        Arc-length reparametrisation: v = v_max * (dp/du) / |dp/du|
        => |v| = v_max  at all t, even where curvature varies.

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        tangent = np.array([
            self.amplitude * np.cos(u),
            self.amplitude * np.cos(2 * u),   # d/du[sin(u)*cos(u)] = cos(2u)
            0,
        ])
        tangent_mag = np.linalg.norm(tangent)
        if tangent_mag > 1e-9:
            tangent = tangent / tangent_mag
        return self.max_velocity * tangent

    def position_func(self, t: float) -> np.ndarray:
        """Parametric lemniscate position.

        p(t) = (A*sin u, A*sin u * cos u, z),  u = 2π*t/T

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        return np.array([
            self.amplitude * np.sin(u),
            self.amplitude * np.sin(u) * np.cos(u),
            2.0,
        ])


class Moon(Trajectory):
    """3D crescent/moon-shaped path at constant forward speed.

    Trajectory design (Myrrh):
    - Like Figure8/Circle but in 3D space
    - Net velocity remains constant while XYZ vary
    - Path resembles crescent moon on a 3D sphere

    Implementation (Copilot):
    - Parametric 3D curve: spherical lemniscate
    - position_func returns pure sphere coords (no altitude offset)
    - Flight altitude z=2.0 added in get_state so velocity stays consistent with position
    - Acceleration via central-difference on position_func, same as Figure8
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0, radius: float = 1.5):
        """Initialize Moon trajectory.

        Args:
            duration: Total time
            max_velocity: Constant forward speed
            radius: Radius of 3D sphere hosting the curve

        Design written by: Myrrh
        Implementation written by: Copilot
        """
        self.radius = radius
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Second derivative of position via central difference.

        a(t) ≈ (p(t+ε) - 2p(t) + p(t-ε)) / ε²

        Written by: Copilot
        """
        eps = 0.0001
        t_lo = max(t - eps, 0)
        t_hi = min(t + eps, self.duration)
        return (
            self.position_func(t_hi)
            - 2 * self.position_func(t)
            + self.position_func(t_lo)
        ) / (eps ** 2)

    def velocity_func(self, t: float) -> np.ndarray:
        """Tangent to 3D lemniscate, scaled to constant speed.

        Raw tangent: dp/du = R*(cos u, cos 2u, -0.5*sin u)
        Arc-length reparametrisation: v = v_max * (dp/du) / |dp/du|

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        tangent = np.array([
            self.radius * np.cos(u),
            self.radius * np.cos(2 * u),
            -self.radius * 0.5 * np.sin(u),
        ])
        tangent_mag = np.linalg.norm(tangent)
        if tangent_mag > 1e-9:
            tangent = tangent / tangent_mag
        return self.max_velocity * tangent

    def position_func(self, t: float) -> np.ndarray:
        """Pure spherical lemniscate position (no altitude offset).

        p(t) = (R*sin u, R*sin u * cos u, R*0.5*cos u),  u = 2π*t/T

        Altitude offset z=2.0 is applied in get_state, keeping velocity_func
        and position_func consistent (d/dt of position_func = velocity_func).

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        return np.array([
            self.radius * np.sin(u),
            self.radius * np.sin(u) * np.cos(u),
            self.radius * 0.5 * np.cos(u),
        ])
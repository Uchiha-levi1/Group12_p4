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

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation.
    
    Written by: Copilot
    """
    duration: float = 10.0  # Total trajectory duration in seconds
    max_velocity: float = 1.0  # Peak velocity (line/square) or constant speed (curves)


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

        # Pre-compute all functions at 1000 Hz and cache as splines
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
            a = self.acceleration_func(t)
            v = self.velocity_func(t)
            p = self.position_func(t)

            accels.append(a)
            velocities.append(v)
            positions.append(p)

        # Convert to numpy arrays for spline fitting
        accels = np.array(accels)
        velocities = np.array(velocities)
        positions = np.array(positions)

        # Store as cubic spline interpolants for fast O(1) lookup
        self.accel_interp = CubicSpline(times, accels, axis=0)
        self.vel_interp = CubicSpline(times, velocities, axis=0)
        self.pos_interp = CubicSpline(times, positions, axis=0)

    @abstractmethod
    def acceleration_func(self, t: float) -> np.ndarray:
        """Return acceleration (ax, ay, az) at time t.

        Subclass implements specific trajectory shape.
        """
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

        Written by: Copilot
        """
        # Clamp time to valid range
        t = np.clip(t, 0, self.duration - 0.001)

        p = self.pos_interp(t)
        v = self.vel_interp(t)
        a = self.accel_interp(t)

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
            # Stationary: return identity quaternion
            return np.array([0, 0, 0, 1])  # [x, y, z, w] format

        # Yaw: angle of velocity in x-y plane
        yaw = np.arctan2(velocity[1], velocity[0])

        # Small roll/pitch tilt based on acceleration
        # (proportional damping to avoid excessive tilt)
        pitch = np.arcsin(np.clip(accel[2] / (9.81 + 1e-9), -0.3, 0.3)) * 0.1
        roll = np.arcsin(np.clip(-accel[1] / (9.81 + 1e-9), -0.3, 0.3)) * 0.1

        # Convert Euler (roll, pitch, yaw) to quaternion
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
        # Duration breakdown: segment1 + turn + segment2
        self.seg_duration = (duration - 1.0) / 2  # Each segment ~4.5 sec if duration=10
        self.turn_start = self.seg_duration
        self.turn_end = self.seg_duration + 1.0
        
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Linear acceleration ramp up then down.

        Written by: Copilot (implementation of Myrrh's design)
        """
        if t < self.turn_start:
            # Segment 1: forward motion
            a_mag = self._linear_ramp_accel(t, self.seg_duration)
            return np.array([a_mag, 0, 0])

        elif t < self.turn_end:
            # Turn: stationary, no acceleration
            return np.array([0, 0, 0])

        else:
            # Segment 2: return motion
            seg_t = t - self.turn_end
            a_mag = self._linear_ramp_accel(seg_t, self.seg_duration)
            return np.array([-a_mag, 0, 0])  # Negative direction

    def velocity_func(self, t: float) -> np.ndarray:
        """Quadratic velocity profile (integrating linear acceleration).

        Written by: Copilot
        """
        if t < self.turn_start:
            # Segment 1: forward
            v_mag = self._linear_ramp_velocity(t, self.seg_duration)
            return np.array([v_mag, 0, 0])

        elif t < self.turn_end:
            # Turn: velocity = 0
            return np.array([0, 0, 0])

        else:
            # Segment 2: return
            seg_t = t - self.turn_end
            v_mag = self._linear_ramp_velocity(seg_t, self.seg_duration)
            return np.array([-v_mag, 0, 0])

    def position_func(self, t: float) -> np.ndarray:
        """Cubic position profile (integrating quadratic velocity).

        Written by: Copilot
        """
        z = 2.0  # Constant altitude (down-facing camera)

        if t < self.turn_start:
            # Segment 1: forward
            x = self._linear_ramp_position(t, self.seg_duration)
            return np.array([x, 0, z])

        elif t < self.turn_end:
            # Turn: stationary at end of segment 1
            x_max = self._linear_ramp_position(self.seg_duration, self.seg_duration)
            return np.array([x_max, 0, z])

        else:
            # Segment 2: return toward origin
            seg_t = t - self.turn_end
            delta_x = self._linear_ramp_position(seg_t, self.seg_duration)
            x_max = self._linear_ramp_position(self.seg_duration, self.seg_duration)
            return np.array([x_max - delta_x, 0, z])

    def _linear_ramp_accel(self, t: float, seg_dur: float) -> float:
        """Linear acceleration: ramp up first half, down second half.

        Written by: Copilot
        """
        half = seg_dur / 2

        if t < half:
            # Linear ramp up: a(t) = a_peak * (t / half)
            a_peak = 2 * self.max_velocity / seg_dur
            return a_peak * (t / half)
        else:
            # Linear ramp down: a(t) = a_peak * (1 - (t - half) / half)
            a_peak = 2 * self.max_velocity / seg_dur
            return a_peak * (1 - (t - half) / half)

    def _linear_ramp_velocity(self, t: float, seg_dur: float) -> float:
        """Integrate linear acceleration to get quadratic velocity.

        v(t) = ∫a(t)dt

        Written by: Copilot
        """
        half = seg_dur / 2

        if t < half:
            # Accel phase: v(t) = (v_max / 2) * (t / half)²
            return (self.max_velocity / 2) * (t / half) ** 2
        else:
            # Decel phase: v(t) = v_max - (v_max / 2) * ((t - half) / half)²
            return self.max_velocity - (self.max_velocity / 2) * ((t - half) / half) ** 2

    def _linear_ramp_position(self, t: float, seg_dur: float) -> float:
        """Integrate quadratic velocity to get cubic position.

        p(t) = ∫v(t)dt

        Written by: Copilot
        """
        half = seg_dur / 2

        if t < half:
            # Accel phase: p(t) = (v_max / 6) * (t / half)³
            return (self.max_velocity / 6) * (t / half) ** 3
        else:
            # Decel phase: p(t) = v_max*half/2 - (v_max / 6) * ((t - half) / half)³
            return (self.max_velocity * half / 2) - (self.max_velocity / 6) * (
                (t - half) / half
            ) ** 3


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
        # Each cycle: segment + turn
        self.cycle_duration = (duration - 4.0) / 4  # 4 turns = 4 seconds, divided into 4 cycles
        self.seg_duration = self.cycle_duration - 1.0
        self.turn_duration = 1.0
        
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Cycle through 4 segments with linear ramp each.

        Written by: Copilot
        """
        cycle_time = self.seg_duration + self.turn_duration
        cycle_num = int(t / cycle_time)
        t_in_cycle = t - cycle_num * cycle_time

        # Directions for each side: +x, +y, -x, -y
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = directions[cycle_num % 4]

        if t_in_cycle < self.seg_duration:
            # Motion phase: linear ramp
            a_mag = self._linear_ramp_accel(t_in_cycle, self.seg_duration)
            return np.array([a_mag * dx, a_mag * dy, 0])
        else:
            # Turn phase: no acceleration
            return np.array([0, 0, 0])

    def velocity_func(self, t: float) -> np.ndarray:
        """Cycle through 4 segments with quadratic velocity.

        Written by: Copilot
        """
        cycle_time = self.seg_duration + self.turn_duration
        cycle_num = int(t / cycle_time)
        t_in_cycle = t - cycle_num * cycle_time

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = directions[cycle_num % 4]

        if t_in_cycle < self.seg_duration:
            v_mag = self._linear_ramp_velocity(t_in_cycle, self.seg_duration)
            return np.array([v_mag * dx, v_mag * dy, 0])
        else:
            return np.array([0, 0, 0])

    def position_func(self, t: float) -> np.ndarray:
        """Position integrating through square perimeter.

        Written by: Copilot
        """
        z = 2.0

        cycle_time = self.seg_duration + self.turn_duration
        cycle_num = int(t / cycle_time)
        t_in_cycle = t - cycle_num * cycle_time

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = directions[cycle_num % 4]

        # Distance per segment
        if t_in_cycle < self.seg_duration:
            seg_dist = self._linear_ramp_position(t_in_cycle, self.seg_duration)
        else:
            seg_dist = self._linear_ramp_position(self.seg_duration, self.seg_duration)

        # Cumulative distance from previous segments
        seg_distance = self._linear_ramp_position(self.seg_duration, self.seg_duration)
        cumulative_dist = cycle_num * seg_distance

        # Current position along the square
        current_dist = cumulative_dist + seg_dist

        return np.array([dx * current_dist, dy * current_dist, z])

    def _linear_ramp_accel(self, t: float, seg_dur: float) -> float:
        """Linear acceleration ramp.

        Written by: Copilot
        """
        half = seg_dur / 2

        if t < half:
            a_peak = 2 * self.max_velocity / seg_dur
            return a_peak * (t / half)
        else:
            a_peak = 2 * self.max_velocity / seg_dur
            return a_peak * (1 - (t - half) / half)

    def _linear_ramp_velocity(self, t: float, seg_dur: float) -> float:
        """Quadratic velocity from linear acceleration.

        Written by: Copilot
        """
        half = seg_dur / 2

        if t < half:
            return (self.max_velocity / 2) * (t / half) ** 2
        else:
            return self.max_velocity - (self.max_velocity / 2) * ((t - half) / half) ** 2

    def _linear_ramp_position(self, t: float, seg_dur: float) -> float:
        """Cubic position from quadratic velocity.

        Written by: Copilot
        """
        half = seg_dur / 2

        if t < half:
            return (self.max_velocity / 6) * (t / half) ** 3
        else:
            return (self.max_velocity * half / 2) - (self.max_velocity / 6) * (
                (t - half) / half
            ) ** 3


class Circle(Trajectory):
    """Circular path at constant forward speed.

    Trajectory design (Myrrh):
    - Forward velocity is constant throughout
    - XYZ change to follow circle path
    - Completes full circle over trajectory duration

    Implementation (Copilot):
    - Parametric circle: x = R*cos(θ), y = R*sin(θ)
    - Velocity tangent to circle with constant magnitude
    - Acceleration is centripetal (perpendicular, pointing toward center)
    """

    def __init__(self, duration: float = 10.0, max_velocity: float = 1.0, radius: float = 2.0):
        """Initialize Circle trajectory.

        Args:
            duration: Total time to complete circle
            max_velocity: Constant forward speed
            radius: Circle radius in meters

        Design written by: Myrrh
        Implementation written by: Copilot
        """
        self.radius = radius
        super().__init__(duration, max_velocity)

    def acceleration_func(self, t: float) -> np.ndarray:
        """Centripetal acceleration pointing toward center.

        a = v² / R, directed toward center

        Written by: Copilot
        """
        theta = (t / self.duration) * 2 * np.pi

        # Centripetal acceleration magnitude
        a_mag = (self.max_velocity ** 2) / self.radius

        # Direction: toward center of circle (negative of radial direction)
        a_x = -a_mag * np.cos(theta)
        a_y = -a_mag * np.sin(theta)

        return np.array([a_x, a_y, 0])

    def velocity_func(self, t: float) -> np.ndarray:
        """Tangent to circle with constant speed.

        Written by: Copilot
        """
        theta = (t / self.duration) * 2 * np.pi
        dtheta_dt = 2 * np.pi / self.duration

        # Tangent vector (perpendicular to radius)
        v_x = -self.radius * np.sin(theta) * dtheta_dt
        v_y = self.radius * np.cos(theta) * dtheta_dt

        return np.array([v_x, v_y, 0])

    def position_func(self, t: float) -> np.ndarray:
        """Parametric circle position.

        Written by: Copilot
        """
        theta = (t / self.duration) * 2 * np.pi
        z = 2.0

        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)

        return np.array([x, y, z])


class Figure8(Trajectory):
    """Figure-8 (lemniscate) path at constant forward speed.

    Trajectory design (Myrrh):
    - Forward velocity is constant throughout
    - XYZ change to follow figure-8 path
    - More complex than circle, loops twice

    Implementation (Copilot):
    - Parametric lemniscate: x = A*sin(u), y = A*sin(u)*cos(u)
    - Velocity tangent with constant magnitude
    - Acceleration perpendicular (steering/curvature effect)
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
        """Curvature-based acceleration (steering toward path center).

        Written by: Copilot
        """
        # Get current velocity direction
        v = self.velocity_func(t)
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-9:
            return np.array([0, 0, 0])

        # Compute curvature by numerical differentiation of velocity
        eps = 0.0001
        v_plus = self.velocity_func(t + eps)
        v_minus = self.velocity_func(t - eps)

        dv_dt = (v_plus - v_minus) / (2 * eps)
        a = dv_dt

        return a

    def velocity_func(self, t: float) -> np.ndarray:
        """Tangent to figure-8 with constant speed.

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        du_dt = 2 * np.pi / self.duration

        # Derivatives of lemniscate parametric form
        # x = A*sin(u)
        # y = A*sin(u)*cos(u)
        dx_du = self.amplitude * np.cos(u)
        dy_du = self.amplitude * (np.cos(2 * u))  # d/du[sin(u)*cos(u)]

        # Tangent vector
        tangent = np.array([dx_du, dy_du, 0])
        tangent_mag = np.linalg.norm(tangent)

        if tangent_mag > 1e-9:
            tangent = tangent / tangent_mag

        # Scale by constant speed
        return self.max_velocity * tangent

    def position_func(self, t: float) -> np.ndarray:
        """Parametric lemniscate position.

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        z = 2.0

        x = self.amplitude * np.sin(u)
        y = self.amplitude * np.sin(u) * np.cos(u)

        return np.array([x, y, z])


class Moon(Trajectory):
    """3D crescent/moon-shaped path at constant forward speed.

    Trajectory design (Myrrh):
    - Like Figure8/Circle but in 3D space
    - Net velocity remains constant while XYZ vary
    - Path resembles crescent moon on a 3D sphere

    Implementation (Copilot):
    - Parametric 3D curve: spherical lemniscate
    - Constant speed along tangent direction
    - Acceleration for curvature following
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
        """Curvature-based acceleration (3D steering).

        Written by: Copilot
        """
        # Numerical differentiation of velocity
        eps = 0.0001
        v_plus = self.velocity_func(t + eps)
        v_minus = self.velocity_func(t - eps)

        a = (v_plus - v_minus) / (2 * eps)
        return a

    def velocity_func(self, t: float) -> np.ndarray:
        """Tangent to 3D lemniscate with constant speed.

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi
        du_dt = 2 * np.pi / self.duration

        # Parametric derivatives (spherical lemniscate)
        # x = R*sin(u)
        # y = R*sin(u)*cos(u)
        # z = R*0.5*cos(u)
        dx_du = self.radius * np.cos(u)
        dy_du = self.radius * np.cos(2 * u)
        dz_du = -self.radius * 0.5 * np.sin(u)

        tangent = np.array([dx_du, dy_du, dz_du])
        tangent_mag = np.linalg.norm(tangent)

        if tangent_mag > 1e-9:
            tangent = tangent / tangent_mag

        return self.max_velocity * tangent

    def position_func(self, t: float) -> np.ndarray:
        """Parametric 3D lemniscate (spherical crescent).

        Written by: Copilot
        """
        u = (t / self.duration) * 2 * np.pi

        x = self.radius * np.sin(u)
        y = self.radius * np.sin(u) * np.cos(u)
        z = 2.0 + 0.5 * self.radius * np.cos(u)  # Offset for altitude

        return np.array([x, y, z])

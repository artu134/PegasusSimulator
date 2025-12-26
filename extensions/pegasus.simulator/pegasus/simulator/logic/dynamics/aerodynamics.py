"""
| File: aerodynamics.py
| Author: Pegasus Simulator Team
| Description: Computes aerodynamic forces (lift, drag) and moments for fixed-wing aircraft
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
"""
import numpy as np
from pegasus.simulator.logic.state import State


class Aerodynamics:
    """
    Class that implements aerodynamic force and moment computations for fixed-wing aircraft.
    Uses a simplified linear aerodynamic model valid for small angles of attack.
    """

    def __init__(self, config={}):
        """
        Initialize the Aerodynamics model with aircraft parameters.

        Args:
            config (dict): Configuration dictionary with aerodynamic parameters.
        
        Examples:
            Default parameters are for a small RC-style aircraft (~2m wingspan):

            >>> {
            >>>     "wing_area": 0.5,           # Wing area [m^2]
            >>>     "wing_span": 2.0,           # Wing span [m]
            >>>     "mean_chord": 0.25,         # Mean aerodynamic chord [m]
            >>>     "air_density": 1.225,       # Air density at sea level [kg/m^3]
            >>>     "CL0": 0.2,                 # Lift coefficient at zero AoA
            >>>     "CL_alpha": 5.7,            # Lift curve slope [1/rad]
            >>>     "CD0": 0.02,                # Parasitic drag coefficient
            >>>     "k": 0.04,                  # Induced drag factor (CD = CD0 + k*CL^2)
            >>>     "CM0": 0.0,                 # Pitching moment at zero AoA
            >>>     "CM_alpha": -0.5,           # Pitching moment slope [1/rad]
            >>>     "CL_max": 1.2,              # Maximum lift coefficient (stall limit)
            >>>     "CL_min": -0.5,             # Minimum lift coefficient
            >>> }
        """
        # Wing geometry parameters
        self._wing_area = config.get("wing_area", 0.5)           # m^2
        self._wing_span = config.get("wing_span", 2.0)           # m
        self._mean_chord = config.get("mean_chord", 0.25)        # m
        
        # Atmospheric parameters
        self._air_density = config.get("air_density", 1.225)     # kg/m^3 at sea level
        
        # Lift coefficients
        self._CL0 = config.get("CL0", 0.2)                       # Lift at zero AoA
        self._CL_alpha = config.get("CL_alpha", 5.7)             # Lift curve slope [1/rad]
        self._CL_max = config.get("CL_max", 1.2)                 # Stall limit
        self._CL_min = config.get("CL_min", -0.5)                # Negative stall limit
        
        # Drag coefficients (parabolic drag polar: CD = CD0 + k*CL^2)
        self._CD0 = config.get("CD0", 0.02)                      # Parasitic drag
        self._k = config.get("k", 0.04)                          # Induced drag factor
        
        # Pitching moment coefficients
        self._CM0 = config.get("CM0", 0.0)                       # Moment at zero AoA
        self._CM_alpha = config.get("CM_alpha", -0.5)            # Moment slope [1/rad]
        
        # Side force and yaw/roll moment coefficients (simplified)
        self._CY_beta = config.get("CY_beta", -0.3)              # Side force due to sideslip
        self._Cl_beta = config.get("Cl_beta", -0.05)             # Roll moment due to sideslip
        self._Cn_beta = config.get("Cn_beta", 0.05)              # Yaw moment due to sideslip
        
        # Damping derivatives (angular rate effects)
        self._Cl_p = config.get("Cl_p", -0.5)                    # Roll damping
        self._Cm_q = config.get("Cm_q", -10.0)                   # Pitch damping
        self._Cn_r = config.get("Cn_r", -0.1)                    # Yaw damping
        
        # Output forces and moments (in body frame)
        self._lift_force = 0.0
        self._drag_force = 0.0
        self._side_force = 0.0
        self._aero_force = np.array([0.0, 0.0, 0.0])
        self._aero_moment = np.array([0.0, 0.0, 0.0])
        
        # State variables for debugging/logging
        self._alpha = 0.0  # Angle of attack [rad]
        self._beta = 0.0   # Sideslip angle [rad]
        self._airspeed = 0.0  # True airspeed [m/s]

    @property
    def lift_force(self):
        """Current lift force magnitude [N]"""
        return self._lift_force

    @property
    def drag_force(self):
        """Current drag force magnitude [N]"""
        return self._drag_force

    @property
    def aero_force(self):
        """Aerodynamic force vector in body frame [N]"""
        return self._aero_force

    @property
    def aero_moment(self):
        """Aerodynamic moment vector in body frame [Nm]"""
        return self._aero_moment

    @property
    def alpha(self):
        """Current angle of attack [rad]"""
        return self._alpha

    @property
    def beta(self):
        """Current sideslip angle [rad]"""
        return self._beta

    @property
    def airspeed(self):
        """Current true airspeed [m/s]"""
        return self._airspeed

    def compute_alpha_beta(self, body_velocity):
        """
        Compute angle of attack and sideslip angle from body velocity.

        Args:
            body_velocity (np.array): Velocity vector in body frame [u, v, w] [m/s]
        
        Returns:
            tuple: (alpha, beta) in radians
        """
        u, v, w = body_velocity[0], body_velocity[1], body_velocity[2]
        
        # Compute airspeed
        airspeed = np.linalg.norm(body_velocity)
        
        if airspeed < 0.1:  # Avoid division by zero at very low speeds
            return 0.0, 0.0
        
        # Angle of attack: alpha = atan2(w, u)
        # Note: In FLU frame, positive w is up, positive u is forward
        if abs(u) > 0.01:
            alpha = np.arctan2(w, u)
        else:
            alpha = 0.0
        
        # Sideslip angle: beta = asin(v / V)
        beta = np.arcsin(np.clip(v / airspeed, -1.0, 1.0))
        
        return alpha, beta

    def update(self, state: State, dt: float):
        """
        Compute aerodynamic forces and moments based on current vehicle state.

        Args:
            state (State): The current state of the vehicle.
            dt (float): Time step [s] (not used in steady-state aero model).

        Returns:
            tuple: (force_body, moment_body) - forces and moments in body frame
        """
        # Get body velocity (FLU frame)
        body_vel = state.linear_body_velocity
        self._airspeed = np.linalg.norm(body_vel)
        
        # Compute angles
        self._alpha, self._beta = self.compute_alpha_beta(body_vel)
        
        # Get angular velocities for damping terms
        p, q, r = state.angular_velocity  # roll, pitch, yaw rates [rad/s]
        
        # Dynamic pressure: q_bar = 0.5 * rho * V^2
        q_bar = 0.5 * self._air_density * self._airspeed ** 2
        
        # Avoid computation at very low airspeeds
        if self._airspeed < 1.0:
            self._lift_force = 0.0
            self._drag_force = 0.0
            self._side_force = 0.0
            self._aero_force = np.array([0.0, 0.0, 0.0])
            self._aero_moment = np.array([0.0, 0.0, 0.0])
            return self._aero_force, self._aero_moment
        
        # ----- Compute Lift Coefficient -----
        CL = self._CL0 + self._CL_alpha * self._alpha
        # Apply stall limits
        CL = np.clip(CL, self._CL_min, self._CL_max)
        
        # ----- Compute Drag Coefficient (parabolic polar) -----
        CD = self._CD0 + self._k * CL ** 2
        
        # ----- Compute Side Force Coefficient -----
        CY = self._CY_beta * self._beta
        
        # ----- Compute Force Magnitudes -----
        self._lift_force = q_bar * self._wing_area * CL
        self._drag_force = q_bar * self._wing_area * CD
        self._side_force = q_bar * self._wing_area * CY
        
        # ----- Convert to Body Frame Forces -----
        # Lift acts perpendicular to velocity (in body z-up direction after rotation by alpha)
        # Drag acts opposite to velocity direction
        cos_alpha = np.cos(self._alpha)
        sin_alpha = np.sin(self._alpha)
        
        # Force in body frame (FLU: x-forward, y-left, z-up)
        # Rotate lift and drag from wind frame to body frame
        Fx = -self._drag_force * cos_alpha + self._lift_force * sin_alpha
        Fy = self._side_force
        Fz = -self._drag_force * sin_alpha - self._lift_force * cos_alpha
        
        self._aero_force = np.array([Fx, Fy, Fz])
        
        # ----- Compute Moment Coefficients -----
        # Non-dimensional angular rates
        if self._airspeed > 1.0:
            p_hat = p * self._wing_span / (2.0 * self._airspeed)
            q_hat = q * self._mean_chord / (2.0 * self._airspeed)
            r_hat = r * self._wing_span / (2.0 * self._airspeed)
        else:
            p_hat = q_hat = r_hat = 0.0
        
        # Roll moment coefficient (includes sideslip and damping)
        Cl = self._Cl_beta * self._beta + self._Cl_p * p_hat
        
        # Pitch moment coefficient (includes AoA and damping)
        Cm = self._CM0 + self._CM_alpha * self._alpha + self._Cm_q * q_hat
        
        # Yaw moment coefficient (includes sideslip and damping)
        Cn = self._Cn_beta * self._beta + self._Cn_r * r_hat
        
        # ----- Compute Moments -----
        Mx = q_bar * self._wing_area * self._wing_span * Cl    # Roll moment
        My = q_bar * self._wing_area * self._mean_chord * Cm   # Pitch moment
        Mz = q_bar * self._wing_area * self._wing_span * Cn    # Yaw moment
        
        self._aero_moment = np.array([Mx, My, Mz])
        
        return self._aero_force, self._aero_moment


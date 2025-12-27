"""
| File: control_surfaces.py
| Author: Pegasus Simulator Team
| Description: Models control surface deflections and resulting aerodynamic moments for fixed-wing aircraft
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pegasus.simulator.logic.state import State


class ControlSurfaces:
    """
    Class that models control surface deflections (aileron, elevator, rudder) 
    and computes the resulting aerodynamic moments.
    
    Supports multiple input modes:
    - PWM mode: Standard servo PWM values (1000-2000)
    - Normalized mode: Normalized values (-1.0 to 1.0)
    - Angle mode: Direct deflection angles in radians
    - Degrees mode: Direct deflection angles in degrees
    - Attitude mode: Target attitude as quaternion or Euler angles
    """

    def __init__(self, config={}):
        """
        Initialize the ControlSurfaces model.

        Args:
            config (dict): Configuration dictionary with control surface parameters.
        
        Examples:
            Default parameters:

            >>> {
            >>>     # PWM configuration
            >>>     "pwm_min": 1000,
            >>>     "pwm_max": 2000,
            >>>     "pwm_center": 1500,
            >>>     
            >>>     # Maximum deflection angles [rad]
            >>>     "aileron_max_deflection": 0.35,   # ~20 degrees
            >>>     "elevator_max_deflection": 0.35,  # ~20 degrees
            >>>     "rudder_max_deflection": 0.35,    # ~20 degrees
            >>>     
            >>>     # Control effectiveness (moment per unit deflection at reference conditions)
            >>>     "aileron_effectiveness": 2.0,    # Roll moment [Nm/rad] per (q_bar * S)
            >>>     "elevator_effectiveness": 3.0,   # Pitch moment [Nm/rad] per (q_bar * S)
            >>>     "rudder_effectiveness": 1.5,     # Yaw moment [Nm/rad] per (q_bar * S)
            >>>     
            >>>     # Wing geometry for moment computation
            >>>     "wing_area": 0.5,       # [m^2]
            >>>     "wing_span": 2.0,       # [m]
            >>>     "mean_chord": 0.25,     # [m]
            >>>     "air_density": 1.225,   # [kg/m^3]
            >>>     
            >>>     # Input mode: "pwm", "normalized", "angle", "degrees", or "attitude"
            >>>     "input_mode": "pwm",
            >>> }
        """
        # PWM configuration
        self._pwm_min = config.get("pwm_min", 1000)
        self._pwm_max = config.get("pwm_max", 2000)
        self._pwm_center = config.get("pwm_center", 1500)
        self._pwm_range = (self._pwm_max - self._pwm_min) / 2.0
        
        # Maximum deflection angles [rad] (~20 degrees each)
        self._aileron_max = config.get("aileron_max_deflection", 0.35)
        self._elevator_max = config.get("elevator_max_deflection", 0.35)
        self._rudder_max = config.get("rudder_max_deflection", 0.35)
        
        # Control effectiveness coefficients (non-dimensional)
        # These represent the change in moment coefficient per radian of deflection
        self._Cl_da = config.get("aileron_effectiveness", 0.15)    # Roll due to aileron
        self._Cm_de = config.get("elevator_effectiveness", -0.5)   # Pitch due to elevator
        self._Cn_dr = config.get("rudder_effectiveness", -0.1)     # Yaw due to rudder
        
        # Cross-coupling effects (optional, usually small)
        self._Cn_da = config.get("aileron_yaw_coupling", 0.01)     # Adverse yaw from aileron
        self._Cl_dr = config.get("rudder_roll_coupling", 0.01)     # Roll from rudder
        
        # Wing geometry
        self._wing_area = config.get("wing_area", 0.5)
        self._wing_span = config.get("wing_span", 2.0)
        self._mean_chord = config.get("mean_chord", 0.25)
        self._air_density = config.get("air_density", 1.225)
        
        # Input mode: "pwm", "normalized", "angle", "degrees", or "attitude"
        self._input_mode = config.get("input_mode", "pwm")
        
        # Attitude control gains (for attitude mode)
        self._roll_gain = config.get("attitude_roll_gain", 1.0)
        self._pitch_gain = config.get("attitude_pitch_gain", 1.0)
        self._yaw_gain = config.get("attitude_yaw_gain", 0.5)
        
        # Current deflections [rad]
        self._aileron_deflection = 0.0
        self._elevator_deflection = 0.0
        self._rudder_deflection = 0.0
        
        # Current PWM inputs
        self._aileron_pwm = self._pwm_center
        self._elevator_pwm = self._pwm_center
        self._rudder_pwm = self._pwm_center
        
        # Target attitude (for attitude mode)
        self._target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # [qx, qy, qz, qw]
        self._target_euler = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw] in rad
        
        # Output moments
        self._control_moment = np.array([0.0, 0.0, 0.0])

    @property
    def aileron_deflection(self):
        """Current aileron deflection [rad]"""
        return self._aileron_deflection

    @property
    def elevator_deflection(self):
        """Current elevator deflection [rad]"""
        return self._elevator_deflection

    @property
    def rudder_deflection(self):
        """Current rudder deflection [rad]"""
        return self._rudder_deflection

    @property
    def control_moment(self):
        """Current control surface moment vector in body frame [Nm]"""
        return self._control_moment

    def pwm_to_normalized(self, pwm):
        """
        Convert PWM value to normalized range [-1, 1].

        Args:
            pwm (float): PWM value (typically 1000-2000)
        
        Returns:
            float: Normalized value in range [-1, 1]
        """
        normalized = (pwm - self._pwm_center) / self._pwm_range
        return np.clip(normalized, -1.0, 1.0)

    def set_control_inputs(self, aileron_pwm, elevator_pwm, rudder_pwm):
        """
        Set control surface positions from PWM values.

        Args:
            aileron_pwm (float): Aileron PWM (1000-2000), 1500 is neutral
            elevator_pwm (float): Elevator PWM (1000-2000), 1500 is neutral
            rudder_pwm (float): Rudder PWM (1000-2000), 1500 is neutral
        """
        self._aileron_pwm = aileron_pwm
        self._elevator_pwm = elevator_pwm
        self._rudder_pwm = rudder_pwm
        
        # Convert PWM to deflection angles
        self._aileron_deflection = self.pwm_to_normalized(aileron_pwm) * self._aileron_max
        self._elevator_deflection = self.pwm_to_normalized(elevator_pwm) * self._elevator_max
        self._rudder_deflection = self.pwm_to_normalized(rudder_pwm) * self._rudder_max

    def set_control_inputs_normalized(self, aileron_norm, elevator_norm, rudder_norm):
        """
        Set control surface positions from normalized values (-1 to 1).

        Args:
            aileron_norm (float): Aileron normalized (-1 to 1)
            elevator_norm (float): Elevator normalized (-1 to 1)
            rudder_norm (float): Rudder normalized (-1 to 1)
        """
        aileron_norm = np.clip(aileron_norm, -1.0, 1.0)
        elevator_norm = np.clip(elevator_norm, -1.0, 1.0)
        rudder_norm = np.clip(rudder_norm, -1.0, 1.0)
        
        self._aileron_deflection = aileron_norm * self._aileron_max
        self._elevator_deflection = elevator_norm * self._elevator_max
        self._rudder_deflection = rudder_norm * self._rudder_max
        
        # Update PWM values for reference
        self._aileron_pwm = self._pwm_center + aileron_norm * self._pwm_range
        self._elevator_pwm = self._pwm_center + elevator_norm * self._pwm_range
        self._rudder_pwm = self._pwm_center + rudder_norm * self._pwm_range

    def set_control_inputs_radians(self, aileron_rad, elevator_rad, rudder_rad):
        """
        Set control surface deflection angles directly in radians.

        Args:
            aileron_rad (float): Aileron deflection [rad]
            elevator_rad (float): Elevator deflection [rad]
            rudder_rad (float): Rudder deflection [rad]
        """
        self._aileron_deflection = np.clip(aileron_rad, -self._aileron_max, self._aileron_max)
        self._elevator_deflection = np.clip(elevator_rad, -self._elevator_max, self._elevator_max)
        self._rudder_deflection = np.clip(rudder_rad, -self._rudder_max, self._rudder_max)
        
        # Update PWM values for reference
        self._aileron_pwm = self._pwm_center + (self._aileron_deflection / self._aileron_max) * self._pwm_range
        self._elevator_pwm = self._pwm_center + (self._elevator_deflection / self._elevator_max) * self._pwm_range
        self._rudder_pwm = self._pwm_center + (self._rudder_deflection / self._rudder_max) * self._pwm_range

    def set_control_inputs_degrees(self, aileron_deg, elevator_deg, rudder_deg):
        """
        Set control surface deflection angles in degrees.

        Args:
            aileron_deg (float): Aileron deflection [degrees]
            elevator_deg (float): Elevator deflection [degrees]
            rudder_deg (float): Rudder deflection [degrees]
        """
        self.set_control_inputs_radians(
            np.radians(aileron_deg),
            np.radians(elevator_deg),
            np.radians(rudder_deg)
        )

    def set_target_attitude_quaternion(self, quaternion):
        """
        Set target attitude as quaternion [qx, qy, qz, qw].
        Control surfaces will be computed to achieve this attitude.

        Args:
            quaternion (array-like): Target quaternion [qx, qy, qz, qw]
        """
        self._target_quaternion = np.array(quaternion)
        # Convert to Euler for reference
        rot = Rotation.from_quat(self._target_quaternion)
        self._target_euler = rot.as_euler('xyz')  # [roll, pitch, yaw]

    def set_target_attitude_euler(self, roll, pitch, yaw, degrees=False):
        """
        Set target attitude as Euler angles.

        Args:
            roll (float): Target roll angle
            pitch (float): Target pitch angle
            yaw (float): Target yaw angle
            degrees (bool): If True, angles are in degrees; otherwise radians
        """
        if degrees:
            roll = np.radians(roll)
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)
        
        self._target_euler = np.array([roll, pitch, yaw])
        rot = Rotation.from_euler('xyz', self._target_euler)
        self._target_quaternion = rot.as_quat()  # [qx, qy, qz, qw]

    def set_inputs(self, aileron, elevator, rudder, mode=None):
        """
        Set control inputs using the specified mode.

        Args:
            aileron (float): Aileron input value
            elevator (float): Elevator input value
            rudder (float): Rudder input value
            mode (str, optional): Input mode - "pwm", "normalized", "angle", or "degrees"
                                  If None, uses the default mode from config.
        """
        input_mode = mode if mode is not None else self._input_mode
        
        if input_mode == "pwm":
            self.set_control_inputs(aileron, elevator, rudder)
        elif input_mode == "normalized":
            self.set_control_inputs_normalized(aileron, elevator, rudder)
        elif input_mode == "angle" or input_mode == "radians":
            self.set_control_inputs_radians(aileron, elevator, rudder)
        elif input_mode == "degrees":
            self.set_control_inputs_degrees(aileron, elevator, rudder)
        else:
            # Default to PWM mode
            self.set_control_inputs(aileron, elevator, rudder)

    def compute_attitude_control(self, current_state: State):
        """
        Compute control surface deflections to achieve target attitude.
        Uses simple proportional control.

        Args:
            current_state (State): Current vehicle state

        Returns:
            tuple: (aileron, elevator, rudder) deflections in radians
        """
        # Get current attitude as Euler angles
        current_rot = Rotation.from_quat(current_state.attitude)
        current_euler = current_rot.as_euler('xyz')  # [roll, pitch, yaw]
        
        # Compute attitude errors
        roll_error = self._target_euler[0] - current_euler[0]
        pitch_error = self._target_euler[1] - current_euler[1]
        yaw_error = self._target_euler[2] - current_euler[2]
        
        # Wrap yaw error to [-pi, pi]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        
        # Simple proportional control
        aileron_cmd = self._roll_gain * roll_error
        elevator_cmd = self._pitch_gain * pitch_error
        rudder_cmd = self._yaw_gain * yaw_error
        
        # Clamp to max deflections
        aileron_cmd = np.clip(aileron_cmd, -self._aileron_max, self._aileron_max)
        elevator_cmd = np.clip(elevator_cmd, -self._elevator_max, self._elevator_max)
        rudder_cmd = np.clip(rudder_cmd, -self._rudder_max, self._rudder_max)
        
        return aileron_cmd, elevator_cmd, rudder_cmd

    def update(self, state: State, dt: float):
        """
        Compute control surface moments based on current deflections and flight state.

        Args:
            state (State): The current state of the vehicle.
            dt (float): Time step [s] (not used directly).

        Returns:
            np.array: Control moment vector [Mx, My, Mz] in body frame [Nm]
        """
        # Get airspeed for dynamic pressure calculation
        airspeed = np.linalg.norm(state.linear_body_velocity)
        
        # Dynamic pressure
        q_bar = 0.5 * self._air_density * airspeed ** 2
        
        # Reference values for moment computation
        q_S = q_bar * self._wing_area
        q_S_b = q_S * self._wing_span    # For roll and yaw
        q_S_c = q_S * self._mean_chord   # For pitch
        
        # Compute moments from control surfaces
        # Roll moment (from aileron, with rudder coupling)
        Mx = q_S_b * (self._Cl_da * self._aileron_deflection + 
                      self._Cl_dr * self._rudder_deflection)
        
        # Pitch moment (from elevator)
        My = q_S_c * self._Cm_de * self._elevator_deflection
        
        # Yaw moment (from rudder, with aileron adverse yaw)
        Mz = q_S_b * (self._Cn_dr * self._rudder_deflection + 
                      self._Cn_da * self._aileron_deflection)
        
        self._control_moment = np.array([Mx, My, Mz])
        
        return self._control_moment

    def reset(self):
        """Reset control surfaces to neutral position."""
        self._aileron_deflection = 0.0
        self._elevator_deflection = 0.0
        self._rudder_deflection = 0.0
        self._aileron_pwm = self._pwm_center
        self._elevator_pwm = self._pwm_center
        self._rudder_pwm = self._pwm_center
        self._target_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self._target_euler = np.array([0.0, 0.0, 0.0])
        self._control_moment = np.array([0.0, 0.0, 0.0])

    @property
    def target_quaternion(self):
        """Target attitude quaternion [qx, qy, qz, qw]"""
        return self._target_quaternion

    @property
    def target_euler(self):
        """Target attitude as Euler angles [roll, pitch, yaw] in radians"""
        return self._target_euler

    @property
    def input_mode(self):
        """Current input mode"""
        return self._input_mode
    
    @input_mode.setter
    def input_mode(self, mode):
        """Set input mode"""
        self._input_mode = mode


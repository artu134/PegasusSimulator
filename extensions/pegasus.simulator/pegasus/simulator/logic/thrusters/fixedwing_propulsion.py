"""
| File: fixedwing_propulsion.py
| Author: Pegasus Simulator Team
| Description: Simple propulsion model for fixed-wing aircraft with a single propeller
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pegasus.simulator.logic.state import State


class FixedWingPropulsion:
    """
    Class that implements a simple propulsion model for fixed-wing aircraft.
    Models a single propeller producing thrust along the body X-axis (forward).
    
    Supports multiple input modes:
    - PWM mode: Standard servo PWM values (1000-2000)
    - Normalized mode: Direct throttle value (0.0 to 1.0)
    - Thrust mode: Direct thrust command in Newtons
    """

    def __init__(self, config={}):
        """
        Initialize the FixedWingPropulsion model.

        Args:
            config (dict): Configuration dictionary with propulsion parameters.
        
        Examples:
            Default parameters for a small RC aircraft:

            >>> {
            >>>     "max_thrust": 15.0,          # Maximum thrust [N]
            >>>     "min_thrust": 0.0,           # Minimum thrust [N]
            >>>     "pwm_min": 1000,             # Minimum PWM value
            >>>     "pwm_max": 2000,             # Maximum PWM value
            >>>     "pwm_arm": 1100,             # PWM value when armed but idle
            >>>     "thrust_tau": 0.1,           # Thrust response time constant [s]
            >>>     "propeller_moment_coeff": 0.01,  # Torque coefficient (propeller reaction torque)
            >>>     "input_mode": "pwm",         # Input mode: "pwm", "normalized", or "thrust"
            >>> }
        """
        # Thrust limits
        self._max_thrust = config.get("max_thrust", 15.0)    # [N]
        self._min_thrust = config.get("min_thrust", 0.0)     # [N]
        
        # PWM configuration
        self._pwm_min = config.get("pwm_min", 1000)
        self._pwm_max = config.get("pwm_max", 2000)
        self._pwm_arm = config.get("pwm_arm", 1100)
        self._pwm_range = self._pwm_max - self._pwm_arm
        
        # Input mode: "pwm", "normalized", or "thrust"
        self._input_mode = config.get("input_mode", "pwm")
        
        # Dynamics
        self._thrust_tau = config.get("thrust_tau", 0.1)     # Time constant [s]
        
        # Propeller reaction torque coefficient
        # Creates a rolling moment opposite to propeller direction
        self._prop_moment_coeff = config.get("propeller_moment_coeff", 0.01)
        
        # Current state
        self._throttle_pwm = self._pwm_min
        self._throttle_normalized = 0.0
        self._commanded_thrust = 0.0
        self._current_thrust = 0.0
        self._propeller_moment = 0.0
        
        # Thrust direction (body frame) - forward along X-axis
        self._thrust_direction = np.array([1.0, 0.0, 0.0])

    @property
    def thrust(self):
        """Current thrust magnitude [N]"""
        return self._current_thrust

    @property
    def thrust_force(self):
        """Current thrust force vector in body frame [N]"""
        return self._thrust_direction * self._current_thrust

    @property
    def propeller_moment(self):
        """Propeller reaction torque [Nm] - creates roll moment"""
        return self._propeller_moment

    @property
    def throttle_normalized(self):
        """Throttle position normalized [0, 1]"""
        return self._throttle_normalized

    def pwm_to_thrust(self, pwm):
        """
        Convert PWM value to thrust command.

        Args:
            pwm (float): Throttle PWM value (1000-2000)
        
        Returns:
            float: Commanded thrust [N]
        """
        # Below arming PWM, no thrust
        if pwm < self._pwm_arm:
            return 0.0
        
        # Linear mapping from PWM to thrust
        normalized = (pwm - self._pwm_arm) / self._pwm_range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        thrust = self._min_thrust + normalized * (self._max_thrust - self._min_thrust)
        return thrust

    def set_throttle(self, throttle_pwm):
        """
        Set throttle from PWM value.

        Args:
            throttle_pwm (float): Throttle PWM (1000-2000)
        """
        self._throttle_pwm = throttle_pwm
        self._throttle_normalized = np.clip((throttle_pwm - self._pwm_arm) / self._pwm_range, 0.0, 1.0)
        self._commanded_thrust = self.pwm_to_thrust(throttle_pwm)

    def set_throttle_normalized(self, throttle_norm):
        """
        Set throttle from normalized value (0.0 to 1.0).

        Args:
            throttle_norm (float): Normalized throttle (0.0 = idle, 1.0 = full)
        """
        self._throttle_normalized = np.clip(throttle_norm, 0.0, 1.0)
        self._throttle_pwm = self._pwm_arm + self._throttle_normalized * self._pwm_range
        self._commanded_thrust = self._min_thrust + self._throttle_normalized * (self._max_thrust - self._min_thrust)

    def set_thrust_direct(self, thrust_newtons):
        """
        Set thrust directly in Newtons.

        Args:
            thrust_newtons (float): Direct thrust command [N]
        """
        self._commanded_thrust = np.clip(thrust_newtons, self._min_thrust, self._max_thrust)
        self._throttle_normalized = (self._commanded_thrust - self._min_thrust) / (self._max_thrust - self._min_thrust)
        self._throttle_pwm = self._pwm_arm + self._throttle_normalized * self._pwm_range

    def set_input(self, value, mode=None):
        """
        Set throttle using the specified input mode.

        Args:
            value (float): Input value (interpretation depends on mode)
            mode (str, optional): Input mode - "pwm", "normalized", or "thrust". 
                                  If None, uses the default mode from config.
        """
        input_mode = mode if mode is not None else self._input_mode
        
        if input_mode == "pwm":
            self.set_throttle(value)
        elif input_mode == "normalized":
            self.set_throttle_normalized(value)
        elif input_mode == "thrust":
            self.set_thrust_direct(value)
        else:
            # Default to PWM mode
            self.set_throttle(value)

    def update(self, state: State, dt: float):
        """
        Update thrust with first-order dynamics (lag).

        Args:
            state (State): The current state of the vehicle (not used in simple model).
            dt (float): Time step [s].

        Returns:
            tuple: (thrust_force, propeller_moment) - thrust vector and reaction torque
        """
        # Apply first-order lag to thrust response
        if self._thrust_tau > 0 and dt > 0:
            alpha = dt / (self._thrust_tau + dt)
            self._current_thrust = (1.0 - alpha) * self._current_thrust + alpha * self._commanded_thrust
        else:
            self._current_thrust = self._commanded_thrust
        
        # Clamp thrust
        self._current_thrust = np.clip(self._current_thrust, self._min_thrust, self._max_thrust)
        
        # Compute propeller reaction torque (creates roll moment)
        # Typically opposes propeller rotation direction
        self._propeller_moment = self._prop_moment_coeff * self._current_thrust
        
        thrust_force = self._thrust_direction * self._current_thrust
        
        return thrust_force, self._propeller_moment

    def reset(self):
        """Reset propulsion to idle state."""
        self._throttle_pwm = self._pwm_min
        self._throttle_normalized = 0.0
        self._commanded_thrust = 0.0
        self._current_thrust = 0.0
        self._propeller_moment = 0.0


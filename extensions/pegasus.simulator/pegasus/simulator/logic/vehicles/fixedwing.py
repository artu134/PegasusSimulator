"""
| File: fixedwing.py
| Author: Pegasus Simulator Team
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the FixedWing class which is used as the base for fixed-wing aircraft vehicles.
"""

import numpy as np
import carb
from scipy.spatial.transform import Rotation

from omni.isaac.dynamic_control import _dynamic_control

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Sensors setup
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

# Aerodynamics and propulsion
from pegasus.simulator.logic.dynamics.aerodynamics import Aerodynamics
from pegasus.simulator.logic.dynamics.control_surfaces import ControlSurfaces
from pegasus.simulator.logic.thrusters.fixedwing_propulsion import FixedWingPropulsion


class FixedWingConfig:
    """
    A data class that is used for configuring a FixedWing aircraft.
    """

    def __init__(self):
        """
        Initialization of the FixedWingConfig class with default values
        for a small RC-style fixed-wing aircraft.
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "fixedwing"

        # The USD file that describes the visual aspect of the vehicle
        self.usd_file = ""

        # ----- Aerodynamics Configuration -----
        self.aerodynamics = Aerodynamics({
            "wing_area": 0.5,           # m^2
            "wing_span": 2.0,           # m
            "mean_chord": 0.25,         # m
            "air_density": 1.225,       # kg/m^3
            "CL0": 0.2,                 # Lift at zero AoA
            "CL_alpha": 5.7,            # Lift curve slope [1/rad]
            "CD0": 0.02,                # Parasitic drag
            "k": 0.04,                  # Induced drag factor
            "CM0": 0.0,                 # Pitch moment at zero AoA
            "CM_alpha": -0.5,           # Pitch moment slope
            "CL_max": 1.2,              # Stall limit
            "CL_min": -0.5,             # Negative stall
        })

        # ----- Control Surfaces Configuration -----
        self.control_surfaces = ControlSurfaces({
            "pwm_min": 1000,
            "pwm_max": 2000,
            "pwm_center": 1500,
            "aileron_max_deflection": 0.35,     # ~20 deg
            "elevator_max_deflection": 0.35,    # ~20 deg
            "rudder_max_deflection": 0.35,      # ~20 deg
            "aileron_effectiveness": 0.15,
            "elevator_effectiveness": -0.5,
            "rudder_effectiveness": -0.1,
            "wing_area": 0.5,
            "wing_span": 2.0,
            "mean_chord": 0.25,
        })

        # ----- Propulsion Configuration -----
        self.propulsion = FixedWingPropulsion({
            "max_thrust": 15.0,         # N - enough for a 2-3 kg aircraft
            "min_thrust": 0.0,
            "pwm_min": 1000,
            "pwm_max": 2000,
            "pwm_arm": 1100,
            "thrust_tau": 0.1,          # Response time constant
            "propeller_moment_coeff": 0.01,
        })

        # ----- Sensors Configuration -----
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]

        # ----- Graphical Sensors -----
        self.graphical_sensors = []

        # ----- Omnigraphs -----
        self.graphs = []

        # ----- Backends Configuration -----
        # By default, no backend - user should configure ArduPilot or custom backend
        self.backends = []
        
        # ----- Control Input Mode -----
        # "pwm" - Standard servo PWM (1000-2000)
        # "normalized" - Normalized values (-1 to 1 for surfaces, 0 to 1 for throttle)
        # "angle" - Direct deflection angles in radians
        # "attitude" - Target attitude (quaternion or euler) + throttle
        self.control_input_mode = "pwm"


class FixedWing(Vehicle):
    """
    FixedWing class - Implements a fixed-wing aircraft vehicle with aerodynamic
    forces, control surfaces, and propulsion.
    """

    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "fixedwing",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.07],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=FixedWingConfig(),
    ):
        """
        Initializes the FixedWing vehicle object.

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator. Defaults to "fixedwing".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position [x, y, z] in ENU convention. Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation as quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (FixedWingConfig): Configuration object. Defaults to FixedWingConfig().
        """

        # Initialize the Vehicle base class
        super().__init__(
            stage_prefix, 
            usd_file, 
            init_pos, 
            init_orientation, 
            config.sensors, 
            config.graphical_sensors, 
            config.graphs, 
            config.backends
        )

        # Store vehicle ID
        self._vehicle_id = vehicle_id

        # Setup aerodynamics, control surfaces, and propulsion
        self._aerodynamics = config.aerodynamics
        self._control_surfaces = config.control_surfaces
        self._propulsion = config.propulsion

        # Control input mapping from ArduPilot/backend
        # Default ArduPlane channel mapping:
        # Channel 0 (servo 1): Aileron
        # Channel 1 (servo 2): Elevator
        # Channel 2 (servo 3): Throttle
        # Channel 3 (servo 4): Rudder
        self._aileron_channel = 0
        self._elevator_channel = 1
        self._throttle_channel = 2
        self._rudder_channel = 3
        
        # Control input mode from config
        self._control_input_mode = config.control_input_mode
        
        # For attitude control mode
        self._target_attitude = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion [qx, qy, qz, qw]
        self._target_euler = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw]
        self._throttle_command = 0.0  # normalized [0, 1]
        self._warned_invalid_force = False

    def start(self):
        """Called when the simulation starts."""
        pass

    def stop(self):
        """Called when the simulation stops."""
        pass

    def update(self, dt: float):
        """
        Method that computes and applies forces to the vehicle based on aerodynamics,
        control surfaces, and propulsion. Called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the articulation root of the vehicle
        articulation = self.get_dc_interface().get_articulation(
            self._stage_prefix + "/body"
        )

        # ----- Get Control Inputs from Backend -----
        if len(self._backends) != 0:
            control_inputs = self._backends[0].input_reference()
            self._process_control_inputs(control_inputs, dt)
        else:
            # No backend - use neutral positions (PWM mode defaults)
            self._control_surfaces.set_control_inputs(1500, 1500, 1500)
            self._propulsion.set_throttle(1000)
        
        # ----- Update Control Surfaces -----
        control_moments = self._control_surfaces.update(self._state, dt)

        # ----- Update Propulsion -----
        thrust_force, prop_moment = self._propulsion.update(self._state, dt)

        # ----- Update Aerodynamics -----
        aero_force, aero_moment = self._aerodynamics.update(self._state, dt)

        # ----- Combine Forces and Moments -----
        # All forces are in body frame (FLU: x-forward, y-left, z-up)
        total_force = aero_force + thrust_force
        total_moment = aero_moment + control_moments
        
        # Add propeller reaction torque (roll moment)
        total_moment[0] += prop_moment

        # Clamp forces/moments to avoid numerical blow-ups
        max_force = 200.0
        max_moment = 50.0
        total_force = np.clip(total_force, -max_force, max_force)
        total_moment = np.clip(total_moment, -max_moment, max_moment)

        # ----- Apply Forces and Torques to the Vehicle Body -----
        if not np.all(np.isfinite(total_force)) or not np.all(np.isfinite(total_moment)):
            if not self._warned_invalid_force:
                carb.log_warn("FixedWing: invalid force/torque detected, skipping physics step")
                self._warned_invalid_force = True
            return

        self.apply_force(
            [total_force[0], total_force[1], total_force[2]], 
            body_part="/body"
        )
        self.apply_torque(
            [total_moment[0], total_moment[1], total_moment[2]], 
            body_part="/body"
        )

        # ----- Handle Propeller Visual (if applicable) -----
        if articulation:
            self.handle_propeller_visual(self._propulsion.throttle_pwm, articulation)

        # ----- Call Backend Updates -----
        for backend in self._backends:
            backend.update(dt)

    def handle_propeller_visual(self, throttle_pwm: float, articulation):
        """
        Auxiliary method to animate the propeller based on throttle setting.

        Args:
            throttle_pwm (float): Current throttle PWM value.
            articulation: The articulation group the propeller joint belongs to.
        """
        # Try to find and animate the propeller joint
        try:
            joint = self.get_dc_interface().find_articulation_dof(articulation, "propeller_joint")
            
            if joint:
                # Scale propeller speed based on throttle
                if throttle_pwm > 1100:
                    # Map throttle to rotation speed
                    normalized_throttle = (throttle_pwm - 1100) / 900.0
                    prop_speed = 50 + normalized_throttle * 200  # 50-250 rad/s
                    self.get_dc_interface().set_dof_velocity(joint, prop_speed)
                else:
                    self.get_dc_interface().set_dof_velocity(joint, 0)
        except:
            # Propeller joint may not exist in simple models
            pass

    def _process_control_inputs(self, control_inputs, dt: float):
        """
        Process control inputs based on the current input mode.

        Args:
            control_inputs: Control inputs from backend (format depends on mode)
            dt (float): Time step
        """
        if self._control_input_mode == "pwm":
            # Standard PWM mode (ArduPilot default)
            if len(control_inputs) >= 4:
                aileron_pwm = control_inputs[self._aileron_channel]
                elevator_pwm = control_inputs[self._elevator_channel]
                throttle_pwm = control_inputs[self._throttle_channel]
                rudder_pwm = control_inputs[self._rudder_channel]
            else:
                aileron_pwm, elevator_pwm, throttle_pwm, rudder_pwm = 1500, 1500, 1000, 1500
            
            self._control_surfaces.set_control_inputs(aileron_pwm, elevator_pwm, rudder_pwm)
            self._propulsion.set_throttle(throttle_pwm)
            
        elif self._control_input_mode == "normalized":
            # Normalized mode: control_inputs = [aileron, elevator, throttle, rudder]
            # Surfaces: -1 to 1, Throttle: 0 to 1
            if len(control_inputs) >= 4:
                aileron_norm = control_inputs[self._aileron_channel]
                elevator_norm = control_inputs[self._elevator_channel]
                throttle_norm = control_inputs[self._throttle_channel]
                rudder_norm = control_inputs[self._rudder_channel]
            else:
                aileron_norm, elevator_norm, throttle_norm, rudder_norm = 0.0, 0.0, 0.0, 0.0
            
            self._control_surfaces.set_control_inputs_normalized(aileron_norm, elevator_norm, rudder_norm)
            self._propulsion.set_throttle_normalized(throttle_norm)
            
        elif self._control_input_mode == "angle":
            # Direct angle mode: control_inputs = [aileron_rad, elevator_rad, throttle_norm, rudder_rad]
            if len(control_inputs) >= 4:
                aileron_rad = control_inputs[self._aileron_channel]
                elevator_rad = control_inputs[self._elevator_channel]
                throttle_norm = control_inputs[self._throttle_channel]
                rudder_rad = control_inputs[self._rudder_channel]
            else:
                aileron_rad, elevator_rad, throttle_norm, rudder_rad = 0.0, 0.0, 0.0, 0.0
            
            self._control_surfaces.set_control_inputs_radians(aileron_rad, elevator_rad, rudder_rad)
            self._propulsion.set_throttle_normalized(throttle_norm)
            
        elif self._control_input_mode == "attitude":
            # Attitude control mode: control_inputs = [qx, qy, qz, qw, throttle_norm]
            # or [roll, pitch, yaw, throttle_norm] for euler
            if len(control_inputs) >= 5:
                # Quaternion mode: [qx, qy, qz, qw, throttle]
                self._target_attitude = np.array(control_inputs[0:4])
                self._throttle_command = control_inputs[4]
                self._control_surfaces.set_target_attitude_quaternion(self._target_attitude)
            elif len(control_inputs) >= 4:
                # Euler mode: [roll, pitch, yaw, throttle]
                self._target_euler = np.array(control_inputs[0:3])
                self._throttle_command = control_inputs[3]
                self._control_surfaces.set_target_attitude_euler(
                    self._target_euler[0], self._target_euler[1], self._target_euler[2]
                )
            else:
                self._throttle_command = 0.0
            
            # Compute control surface commands from attitude error
            aileron_cmd, elevator_cmd, rudder_cmd = self._control_surfaces.compute_attitude_control(self._state)
            self._control_surfaces.set_control_inputs_radians(aileron_cmd, elevator_cmd, rudder_cmd)
            self._propulsion.set_throttle_normalized(self._throttle_command)
        else:
            # Default to PWM mode
            if len(control_inputs) >= 4:
                self._control_surfaces.set_control_inputs(
                    control_inputs[0], control_inputs[1], control_inputs[3]
                )
                self._propulsion.set_throttle(control_inputs[2])

    def set_control_input_mode(self, mode: str):
        """
        Set the control input mode.

        Args:
            mode (str): Input mode - "pwm", "normalized", "angle", or "attitude"
        """
        self._control_input_mode = mode

    def set_target_attitude(self, quaternion=None, euler=None, throttle=0.0, degrees=False):
        """
        Set target attitude for attitude control mode.

        Args:
            quaternion (array-like, optional): Target quaternion [qx, qy, qz, qw]
            euler (array-like, optional): Target Euler angles [roll, pitch, yaw]
            throttle (float): Throttle command (0.0 to 1.0)
            degrees (bool): If True, euler angles are in degrees
        """
        if quaternion is not None:
            self._target_attitude = np.array(quaternion)
            self._control_surfaces.set_target_attitude_quaternion(quaternion)
        elif euler is not None:
            if degrees:
                euler = np.radians(euler)
            self._target_euler = np.array(euler)
            self._control_surfaces.set_target_attitude_euler(euler[0], euler[1], euler[2])
        
        self._throttle_command = np.clip(throttle, 0.0, 1.0)

    def set_direct_control(self, aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.0, mode="normalized"):
        """
        Set control inputs directly (bypassing backend).
        Useful for testing or custom control implementations.

        Args:
            aileron (float): Aileron command
            elevator (float): Elevator command
            rudder (float): Rudder command
            throttle (float): Throttle command
            mode (str): Input mode - "pwm", "normalized", "angle", or "degrees"
        """
        if mode == "pwm":
            self._control_surfaces.set_control_inputs(aileron, elevator, rudder)
            self._propulsion.set_throttle(throttle)
        elif mode == "normalized":
            self._control_surfaces.set_control_inputs_normalized(aileron, elevator, rudder)
            self._propulsion.set_throttle_normalized(throttle)
        elif mode == "angle" or mode == "radians":
            self._control_surfaces.set_control_inputs_radians(aileron, elevator, rudder)
            self._propulsion.set_throttle_normalized(throttle)
        elif mode == "degrees":
            self._control_surfaces.set_control_inputs_degrees(aileron, elevator, rudder)
            self._propulsion.set_throttle_normalized(throttle)

    @property
    def control_input_mode(self):
        """Current control input mode"""
        return self._control_input_mode

    @property
    def target_attitude(self):
        """Target attitude quaternion [qx, qy, qz, qw]"""
        return self._target_attitude

    @property
    def target_euler(self):
        """Target Euler angles [roll, pitch, yaw] in radians"""
        return self._target_euler

    @property
    def aerodynamics(self):
        """Access to the aerodynamics model."""
        return self._aerodynamics

    @property
    def control_surfaces(self):
        """Access to the control surfaces model."""
        return self._control_surfaces

    @property
    def propulsion(self):
        """Access to the propulsion model."""
        return self._propulsion

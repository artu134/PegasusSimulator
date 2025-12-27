#!/usr/bin/env python
"""
| File: 13_fixedwing_simple_test.py
| Author: Pegasus Simulator Team
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Simple test flight for fixed-wing aircraft using a custom Python controller.
|              No ArduPilot required - just run this script to see the aircraft fly!
|
| Usage:
|   cd ~/Desktop/PegasusSimulator/examples
|   python 13_fixedwing_simple_test.py
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

# Import the Pegasus API
from pegasus.simulator.params import ROBOTS
from pxr import Gf, UsdGeom, UsdLux
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.viewports import set_camera_view
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.vehicles.fixedwing import FixedWing, FixedWingConfig
from pegasus.simulator.logic.dynamics.aerodynamics import Aerodynamics
from pegasus.simulator.logic.thrusters.fixedwing_propulsion import FixedWingPropulsion
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class SimpleFixedWingController(Backend):
    """
    A simple fixed-wing controller that demonstrates basic flight.
    
    Flight phases:
    1. Takeoff: Full throttle, slight pitch up
    2. Climb: Maintain climb until reaching target altitude
    3. Level off: transition to straight flight
    """

    def __init__(self):
        super().__init__(config=None)
        
        # Control outputs (normalized: -1 to 1 for surfaces, 0 to 1 for throttle)
        self._aileron = 0.0
        self._elevator = 0.0
        self._throttle = 0.0
        self._rudder = 0.0
        
        # Flight parameters
        self._target_altitude = 40.0  # meters
        self._target_airspeed = 12.0  # m/s
        self._cruise_throttle = 0.7
        self._target_pitch_takeoff = np.deg2rad(4.0)
        self._target_pitch_climb = np.deg2rad(3.5)
        self._pitch_gain = 0.65
        self._pitch_damp = 2.0
        self._roll_gain = 1.0
        self._yaw_gain = 0.8
        self._yaw_damp = 0.6
        self._surface_rate = 0.25  # max change per second
        self._takeoff_speed = 13.0  # m/s
        self._rotate_duration = 5.0  # seconds
        self._rotate_start_time = None
        self._climb_start_time = None
        self._climb_pitch_ramp = 4.0  # seconds
        self._climb_throttle = 0.92
        self._throttle_rate = 0.15  # max change per second
        self._level_start_time = None
        self._level_duration = 6.0  # seconds
        self._level_off_band = 10.0  # meters
        self._climb_pitch_start = None
        self._level_pitch_start = None
        self._climb_stage_time = 6.0  # seconds
        self._climb_pitch_stage = np.deg2rad(2.5)
        self._max_climb_time = 10.0  # seconds
        
        # State
        self._phase = "takeoff"
        self._time = 0.0
        self._turn_start_time = 0.0
        self._turn_direction = 1  # 1 = right, -1 = left
        self._yaw_target = 0.0
        
        # Current state from simulation
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._attitude = np.array([0, 0, 0, 1])
        self._airspeed = 0.0
        
        print("=" * 50)
        print("Simple Fixed-Wing Controller Initialized")
        print("=" * 50)
        print("Flight plan:")
        print(f"  1. Takeoff with full throttle")
        print(f"  2. Climb to {self._target_altitude}m altitude")
        print(f"  3. Level off and fly straight")
        print("=" * 50)

    def start(self):
        """Called when simulation starts."""
        self._phase = "takeoff"
        self._time = 0.0
        self._yaw_target = self._get_yaw_angle()
        self._rotate_start_time = None
        self._climb_start_time = None
        self._climb_pitch_start = None
        self._level_start_time = None
        self._level_pitch_start = None
        print("\n[Controller] Simulation started - Beginning takeoff sequence!")

    def stop(self):
        """Called when simulation stops."""
        print(f"\n[Controller] Flight ended. Total time: {self._time:.1f}s")

    def reset(self):
        """Reset the controller."""
        self._phase = "takeoff"
        self._time = 0.0
        self._climb_pitch_start = None
        self._level_start_time = None
        self._level_pitch_start = None

    def update_sensor(self, sensor_type: str, data):
        """Receive sensor data (not used in this simple controller)."""
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """Receive graphical sensor data (not used)."""
        pass

    def update_state(self, state: State):
        """Receive current vehicle state."""
        self._position = state.position
        self._velocity = state.linear_velocity
        self._attitude = state.attitude
        self._airspeed = np.linalg.norm(state.linear_body_velocity)
        self._angular_velocity = state.angular_velocity

    def input_reference(self):
        """
        Return control inputs as normalized values.
        Format: [aileron, elevator, throttle, rudder]
        """
        # Convert to PWM for compatibility with the default input mode
        aileron_pwm = 1500 + self._aileron * 500
        elevator_pwm = 1500 + self._elevator * 500
        throttle_pwm = 1000 + self._throttle * 1000
        rudder_pwm = 1500 + self._rudder * 500
        
        return [aileron_pwm, elevator_pwm, throttle_pwm, rudder_pwm]

    def update(self, dt: float):
        """Update control logic each timestep."""
        self._time += dt
        
        altitude = self._position[2]
        
        # State machine for flight phases
        if self._phase == "takeoff":
            self._do_takeoff(altitude, dt)
        elif self._phase == "climb":
            self._do_climb(altitude, dt)
        elif self._phase == "level":
            self._do_level(altitude, dt)
        elif self._phase == "cruise":
            self._do_cruise(altitude, dt)
        
        # Log status periodically
        if int(self._time * 2) % 2 == 0 and abs(self._time - int(self._time)) < dt:
            self._log_status(altitude)

    def _do_takeoff(self, altitude, dt):
        """Takeoff phase: full throttle, pitch up."""
        # Throttle ramp to build speed smoothly
        self._throttle = np.clip(0.3 + self._time / 8.0 * 0.7, 0.3, 1.0)

        pitch_target = 0.0
        if self._airspeed >= self._takeoff_speed:
            if self._rotate_start_time is None:
                self._rotate_start_time = self._time
            rotate_progress = (self._time - self._rotate_start_time) / self._rotate_duration
            pitch_target = np.clip(rotate_progress, 0.0, 1.0) * self._target_pitch_takeoff

        elevator_cmd = self._pitch_hold_command(pitch_target, max_elevator=0.3)
        aileron_cmd = self._roll_hold_command()
        rudder_cmd = self._yaw_hold_command()
        self._apply_surface_commands(aileron_cmd, elevator_cmd, rudder_cmd, dt)
        
        # Transition to climb once we have takeoff speed
        if self._airspeed > (self._takeoff_speed + 2.0) and altitude > 1.0:
            self._phase = "climb"
            self._climb_start_time = self._time
            self._climb_pitch_start = self._get_pitch_angle()
            print(f"\n[{self._time:.1f}s] Transitioning to CLIMB phase")

    def _do_climb(self, altitude, dt):
        """Climb phase: maintain climb until target altitude."""
        if self._climb_start_time is None:
            self._climb_start_time = self._time
        if self._climb_pitch_start is None:
            self._climb_pitch_start = self._get_pitch_angle()
        climb_elapsed = max(0.0, self._time - self._climb_start_time)

        # Gradually settle to climb throttle
        airspeed_error = self._target_airspeed - self._airspeed
        throttle_target = self._climb_throttle + np.clip(airspeed_error * 0.025, -0.05, 0.12)
        throttle_target = np.clip(throttle_target, 0.75, 1.0)
        self._throttle = self._slew(self._throttle, throttle_target, self._throttle_rate, dt)

        if climb_elapsed < self._climb_stage_time:
            stage_progress = np.clip(climb_elapsed / self._climb_stage_time, 0.0, 1.0)
            pitch_target = (
                (1.0 - stage_progress) * self._climb_pitch_start
                + stage_progress * self._climb_pitch_stage
            )
        else:
            ramp_elapsed = climb_elapsed - self._climb_stage_time
            climb_progress = np.clip(ramp_elapsed / self._climb_pitch_ramp, 0.0, 1.0)
            pitch_target = (
                (1.0 - climb_progress) * self._climb_pitch_stage
                + climb_progress * self._target_pitch_climb
            )

        altitude_error = max(self._target_altitude - altitude, 0.0)
        level_scale = np.clip(altitude_error / self._level_off_band, 0.0, 1.0)
        pitch_target *= level_scale
        if self._airspeed < self._target_airspeed:
            pitch_target *= 0.6
        if self._velocity[2] > 2.0:
            pitch_target *= 0.5

        elevator_cmd = self._pitch_hold_command(pitch_target, max_elevator=0.15)
        aileron_cmd = self._roll_hold_command()
        rudder_cmd = self._yaw_hold_command()
        self._apply_surface_commands(aileron_cmd, elevator_cmd, rudder_cmd, dt)
        
        # Transition to cruise at target altitude
        if altitude >= (self._target_altitude - 1.0) or climb_elapsed >= self._max_climb_time:
            self._phase = "level"
            self._level_start_time = self._time
            self._level_pitch_start = self._get_pitch_angle()
            print(f"\n[{self._time:.1f}s] Reached target altitude! Leveling off")

    def _do_level(self, altitude, dt):
        """Level-off phase: smoothly transition to straight flight."""
        if self._level_start_time is None:
            self._level_start_time = self._time
        if self._level_pitch_start is None:
            self._level_pitch_start = self._get_pitch_angle()
        level_elapsed = max(0.0, self._time - self._level_start_time)
        level_progress = np.clip(level_elapsed / self._level_duration, 0.0, 1.0)

        altitude_error = self._target_altitude - altitude
        level_pitch_target = np.clip(altitude_error * 0.02, np.deg2rad(-2.0), np.deg2rad(2.0))
        pitch_target = (
            (1.0 - level_progress) * self._level_pitch_start
            + level_progress * level_pitch_target
        )

        self._throttle = self._slew(self._throttle, self._cruise_throttle, self._throttle_rate, dt)
        elevator_cmd = self._pitch_hold_command(pitch_target, max_elevator=0.18)
        aileron_cmd = self._roll_hold_command()
        rudder_cmd = self._yaw_hold_command()
        self._apply_surface_commands(aileron_cmd, elevator_cmd, rudder_cmd, dt)

        if level_elapsed >= self._level_duration:
            self._phase = "cruise"
            print(f"\n[{self._time:.1f}s] Level-off complete. Transitioning to CRUISE phase")

    def _do_cruise(self, altitude, dt):
        """Cruise phase: level flight and steady heading."""
        # Throttle for cruise
        self._throttle = self._cruise_throttle
        
        altitude_error = self._target_altitude - altitude
        pitch_target = np.clip(altitude_error * 0.02, np.deg2rad(-4.0), np.deg2rad(4.0))
        elevator_cmd = self._pitch_hold_command(pitch_target, max_elevator=0.2)
        aileron_cmd = self._roll_hold_command()
        rudder_cmd = self._yaw_hold_command()
        self._apply_surface_commands(aileron_cmd, elevator_cmd, rudder_cmd, dt)

    def _get_roll_angle(self):
        """Get current roll angle in radians."""
        rot = Rotation.from_quat(self._attitude)
        euler = rot.as_euler('xyz')
        return euler[0]

    def _get_pitch_angle(self):
        """Get current pitch angle in radians."""
        rot = Rotation.from_quat(self._attitude)
        euler = rot.as_euler('xyz')
        return euler[1]

    def _get_yaw_angle(self):
        """Get current yaw angle in radians."""
        rot = Rotation.from_quat(self._attitude)
        euler = rot.as_euler('xyz')
        return euler[2]

    def _roll_hold_command(self):
        """Simple roll stabilization using aileron."""
        roll = self._get_roll_angle()
        return np.clip(-self._roll_gain * roll, -0.2, 0.2)

    def _pitch_hold_command(self, target_pitch, max_elevator):
        """Pitch-hold controller with pitch-rate damping."""
        pitch_error = target_pitch - self._get_pitch_angle()
        pitch_rate = self._angular_velocity[1]
        command = (self._pitch_gain * pitch_error) - (self._pitch_damp * pitch_rate)
        return np.clip(-command, -max_elevator, max_elevator)

    def _yaw_hold_command(self):
        """Yaw-hold controller with yaw-rate damping."""
        yaw_error = self._wrap_angle(self._yaw_target - self._get_yaw_angle())
        yaw_rate = self._angular_velocity[2]
        command = (self._yaw_gain * yaw_error) - (self._yaw_damp * yaw_rate)
        return np.clip(command, -0.2, 0.2)

    def _apply_surface_commands(self, aileron_cmd, elevator_cmd, rudder_cmd, dt):
        """Apply slew-rate limiting to surface commands."""
        self._aileron = self._slew(self._aileron, aileron_cmd, self._surface_rate, dt)
        self._elevator = self._slew(self._elevator, elevator_cmd, self._surface_rate, dt)
        self._rudder = self._slew(self._rudder, rudder_cmd, self._surface_rate, dt)

    def _slew(self, current, target, rate, dt):
        if dt <= 0.0:
            return target
        delta = np.clip(target - current, -rate * dt, rate * dt)
        return current + delta

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


    def _log_status(self, altitude):
        """Print status information."""
        print(f"[{self._time:.1f}s] Phase: {self._phase:8s} | "
              f"Alt: {altitude:5.1f}m | "
              f"Speed: {self._airspeed:4.1f}m/s | "
              f"Throttle: {self._throttle*100:3.0f}%")


class FixedWingTestApp:
    """
    Test application for fixed-wing aircraft.
    """

    def __init__(self):
        # Acquire the timeline
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self._ground_plane = None

        # Add a simple ground plane locally to avoid external asset loads
        self._add_local_ground_plane()
        self._add_basic_lighting()
        self._propeller_prim = None
        self._propeller_rotate_op = None
        self._propeller_angle = 0.0

        # Create the fixed-wing aircraft
        self.vehicle = None
        self.create_aircraft()
        self._set_camera_view()

        # Reset the simulation
        self.world.reset()

        self.stop_sim = False

    def create_aircraft(self):
        """Create a fixed-wing aircraft with simple controller."""
        
        # Create configuration
        config = FixedWingConfig()

        # Boost lift and thrust to ensure takeoff in this simple model
        config.aerodynamics = Aerodynamics({
            "wing_area": 0.9,
            "wing_span": 2.0,
            "mean_chord": 0.3,
            "air_density": 1.225,
            "CL0": 0.1,
            "CL_alpha": 5.0,
            "CD0": 0.03,
            "k": 0.07,
            "CM0": 0.0,
            "CM_alpha": -0.4,
            "CL_max": 1.3,
            "CL_min": -0.6,
            "Cl_p": -1.2,
            "Cm_q": -18.0,
            "Cn_r": -0.3,
        })
        config.propulsion = FixedWingPropulsion({
            "max_thrust": 28.0,
            "min_thrust": 0.0,
            "pwm_min": 1000,
            "pwm_max": 2000,
            "pwm_arm": 1100,
            "thrust_tau": 0.08,
            "propeller_moment_coeff": 0.01,
        })
        
        # Use the custom Python controller
        config.backends = [SimpleFixedWingController()]
        
        # Spawn the aircraft
        # Start facing forward (positive X) with slight nose-up attitude for takeoff
        self.vehicle = FixedWing(
            "/World/fixedwing",
            ROBOTS['FixedWing'],
            0,
            # Start position: on the ground
            [0.0, 0.0, 0.3],
            # Initial orientation: nose-up for easier lift
            Rotation.from_euler("XYZ", [0.0, -5.0, 0.0], degrees=True).as_quat(),
            config=config,
        )
        
        print("\nFixed-wing aircraft spawned!")
        print("Press PLAY in Isaac Sim to start the simulation...")

    def _add_local_ground_plane(self):
        """Create a local ground plane without pulling remote USD assets."""
        if self._ground_plane is not None:
            return
        physics_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/ground_material",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        self._ground_plane = GroundPlane(
            prim_path="/World/ground_plane",
            z_position=0.0,
            color=np.array([0.25, 0.25, 0.25]),
            physics_material=physics_material,
        )
        self.world.scene.add(self._ground_plane)

    def _add_basic_lighting(self):
        """Add simple lighting so the scene isn't black."""
        stage = self.world.stage

        sun_path = "/World/SunLight"
        if not stage.GetPrimAtPath(sun_path).IsValid():
            sun = UsdLux.DistantLight.Define(stage, sun_path)
            sun.CreateIntensityAttr(5000.0)
            sun.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))

        dome_path = "/World/SkyLight"
        if not stage.GetPrimAtPath(dome_path).IsValid():
            dome = UsdLux.DomeLight.Define(stage, dome_path)
            dome.CreateIntensityAttr(500.0)
            dome.CreateColorAttr(Gf.Vec3f(0.7, 0.75, 0.9))

    def _set_camera_view(self):
        """Point the viewport at the aircraft spawn area."""
        self._update_camera()

    def _update_camera(self):
        """Follow the aircraft with the viewport camera."""
        if self.vehicle is None:
            return
        position = self.vehicle.state.position
        eye = [position[0] + 8.0, position[1] - 8.0, position[2] + 4.0]
        target = [position[0], position[1], max(position[2], 0.1)]
        set_camera_view(eye=eye, target=target)

    def _spin_propeller(self, dt):
        """Spin the propeller visually based on throttle."""
        if self.vehicle is None:
            return
        if self._propeller_prim is None:
            self._propeller_prim = self.world.stage.GetPrimAtPath(
                "/World/fixedwing/body/propeller"
            )
            if not self._propeller_prim.IsValid():
                return
            xform = UsdGeom.Xformable(self._propeller_prim)
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    self._propeller_rotate_op = op
                    break
            if self._propeller_rotate_op is None:
                self._propeller_rotate_op = xform.AddRotateXYZOp()

        throttle = getattr(self.vehicle._propulsion, "throttle_normalized", 0.0)
        spin_rate = (30.0 + 270.0 * throttle) * 360.0  # deg/s
        self._propeller_angle = (self._propeller_angle + spin_rate * dt) % 360.0
        self._propeller_rotate_op.Set(Gf.Vec3f(self._propeller_angle, 0.0, 0.0))

    def run(self):
        """Main loop."""
        # Start the simulation
        self.timeline.play()

        print("\n" + "=" * 50)
        print("SIMULATION RUNNING")
        print("=" * 50)
        print("Watch the aircraft take off and fly!")
        print("Press Ctrl+C to stop.")
        print("=" * 50 + "\n")

        # Main loop
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            self._update_camera()
            self._spin_propeller(self.world.get_physics_dt())
        
        # Cleanup
        carb.log_warn("Simulation closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    print("\n" + "=" * 60)
    print("FIXED-WING AIRCRAFT TEST FLIGHT")
    print("=" * 60)
    print()
    print("This script demonstrates a fixed-wing aircraft with a simple")
    print("Python controller. No ArduPilot or external software needed!")
    print()
    print("The aircraft will:")
    print("  1. Take off with full throttle")
    print("  2. Climb to 40 meters altitude")
    print("  3. Level off and fly straight")
    print()
    print("=" * 60 + "\n")
    
    # Create and run the app
    app = FixedWingTestApp()
    app.run()


if __name__ == "__main__":
    main()


